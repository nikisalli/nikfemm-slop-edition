#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdexcept>
#include <string.h>
#include <chrono>
#include <unordered_set>
#include <unordered_map>
#include <iterator>
#include <set>

// before the nikfemm headers, they define macros like PI
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include "../constants.hpp"
#include "simulation.hpp"
#include "../drawing/drawing.hpp"
#include "../geometry/segment.hpp"
#include "../geometry/vector.hpp"
#include "../algebra/coo.hpp"
#include "../algebra/csr.hpp"
#include "../algebra/solvers.hpp"
#include "../algebra/math.hpp"

namespace nikfemm {
    MagnetostaticSystem MagnetostaticSimulation::generateSystem(bool refine, double max_triangle_area, int min_angle, bool refine_magnets) {
        // get time in milliseconds

        mesh.axisymmetric = axisymmetric;
        double coeff = 1.5;  // make the circle [coeff] times the size of the smallest circle
        // spacing of the boundary points
        double boundary_spacing = sqrt((max_triangle_area * 4) / sqrt(3)) / 2;

        if (!axisymmetric) {
            /* auto boundary */
            // find smallest enclosing circle using Welzl's algorithm
            // Circle smallest_circle = Circle::getMinimumEnclosingCircle(mesh.drawing.points);
            Circle smallest_circle = Circle::getEnclosingCircle(mesh.drawing.points);
            if (mesh.drawing.points.size() == 0) {
                smallest_circle.radius = 1;
            }
            // set simulation offset and boundary radius
            mesh.center = smallest_circle.center;
            mesh.radius = smallest_circle.radius * coeff;
            // translate everything to the origin
            mesh.drawing.translate(-mesh.center);
            // make circle double the size of the smallest circle
            Circle boundary_circle = Circle(Vector(0, 0), smallest_circle.radius * coeff);
            uint32_t boundary_points = (uint32_t)(boundary_circle.circumference() / boundary_spacing);
            nloginfo("boundary points: %d", boundary_points);
            mesh.drawing.drawCircle(boundary_circle, boundary_points);
        } else {
            // x is the radius and y the axial coordinate, the symmetry axis is x = 0
            // the boundary is a half circle centered on the axis, closed by the axis itself
            double z_min = std::numeric_limits<double>::infinity();
            double z_max = -std::numeric_limits<double>::infinity();
            for (auto& p : mesh.drawing.points) {
                if (p.x < 0) {
                    nexit("error: axisymmetric problems cannot have points with negative radius (x < 0)");
                }
                z_min = std::min(z_min, p.y);
                z_max = std::max(z_max, p.y);
            }
            mesh.center = mesh.drawing.points.size() ? Vector(0, (z_min + z_max) / 2) : Vector(0, 0);
            double enclosing_radius = 0;
            for (auto& p : mesh.drawing.points) {
                enclosing_radius = std::max(enclosing_radius, Vector::distance(p, mesh.center));
            }
            if (enclosing_radius == 0) {
                enclosing_radius = 1;
            }
            mesh.radius = enclosing_radius * coeff;
            // translate only along the axis, the radius is absolute
            mesh.drawing.translate(-mesh.center);
            double R = mesh.radius;

            // half circle from the bottom to the top of the axis
            uint32_t arc_segments = std::max<uint32_t>(4, PI * R / boundary_spacing);
            nloginfo("boundary points: %d", arc_segments + 1);
            std::vector<Vector> boundary = {Vector(0, -R)};
            for (uint32_t i = 1; i < arc_segments; i++) {
                double angle = -PI / 2 + PI * i / arc_segments;
                boundary.push_back(Vector(R * cos(angle), R * sin(angle)));
            }
            boundary.push_back(Vector(0, R));
            // the axis goes back down through the points of the drawing that lie on it
            std::vector<double> axis_z;
            for (auto& p : mesh.drawing.points) {
                if (p.x == 0) axis_z.push_back(p.y);
            }
            std::sort(axis_z.begin(), axis_z.end(), std::greater<double>());
            axis_z.erase(std::unique(axis_z.begin(), axis_z.end()), axis_z.end());
            for (double z : axis_z) {
                boundary.push_back(Vector(0, z));
            }
            mesh.drawing.drawPolygon(boundary);
        }
        // add region near the edge of the circle
        mesh.drawing.drawRegion(Vector(mesh.radius * 0.9, 0), {0, {0, 0}, magnetostatic_materials::air});
        // add the boundary 
        // mesh.drawing.plot();
        if (refine_magnets) {
            mesh.refineMeshAroundMagnets();
        }

        if (refine) {
            mesh.drawing.addRefiningPoints();
        }

        mesh.mesh(max_triangle_area, min_angle);
        // mesh.plot();
        // the tolerance for geometric comparisons comes from the physical mesh only, the far away kelvin
        // elements can be arbitrarily small or degenerate in the transformed coordinates
        mesh.computeEpsilon();
        mesh.addKelvinBoundaryConditions(axisymmetric);
        // mesh.plot();
        nloginfo("the mesh has %u nodes and %u elements", mesh.data.numberofpoints, mesh.data.numberoftriangles);
        

        auto system = mesh.getFemSystem();
        mesh.addDirichletInfiniteBoundaryConditions(system);
        return system;
    }

    std::vector<double> MagnetostaticSimulation::solve(MagnetostaticSystem& system) {
        typedef Eigen::SparseMatrix<double> SparseMatrix;
        const uint32_t n = system.n;
        const uint32_t m = mesh.data.numberoftriangles;
        const uint32_t nnz = system.row_ind.size();
        auto A = std::vector<double>(n, 0);

        // reluctivity of the linear elements, the nonlinear ones share the smooth BH curve of their material
        std::vector<double> elem_nu(m, 0);
        std::vector<int32_t> elem_spline(m, -1);
        std::vector<BHSpline> splines;
        std::unordered_map<const MagnetostaticProp*, int32_t> spline_ids;
        bool all_linear = true;
        for (uint32_t e = 0; e < m; e++) {
            const MagnetostaticProp* prop = mesh.drawing.getRegionPtrFromId(mesh.data.triangleattributelist[e]);
            if (prop->isLinear()) {
                elem_nu[e] = 1 / prop->mu;
            } else {
                all_linear = false;
                auto it = spline_ids.find(prop);
                if (it == spline_ids.end()) {
                    it = spline_ids.emplace(prop, splines.size()).first;
                    splines.emplace_back(prop->bh_curve);
                }
                elem_spline[e] = it->second;
            }
        }

        // dirichlet nodes keep their value, the newton updates only move the free nodes
        std::vector<bool> fixed(n, false);
        for (uint32_t i = 0; i < system.dirichlet_nodes.size(); i++) {
            fixed[system.dirichlet_nodes[i]] = true;
            A[system.dirichlet_nodes[i]] = system.dirichlet_values[i];
        }
        // entries in the rows and columns of the dirichlet nodes, they are replaced by the identity
        std::vector<int> fixed_entries;
        for (uint32_t j = 0; j < n; j++) {
            for (int k = system.col_ptr[j]; k < system.col_ptr[j + 1]; k++) {
                if (fixed[j] || fixed[system.row_ind[k]]) {
                    fixed_entries.push_back(k);
                }
            }
        }

        // stiffness matrix (the jacobian for nonlinear problems) with the sparsity pattern of the system
        SparseMatrix K(n, n);
        K.resizeNonZeros(nnz);
        std::copy(system.col_ptr.begin(), system.col_ptr.end(), K.outerIndexPtr());
        std::copy(system.row_ind.begin(), system.row_ind.end(), K.innerIndexPtr());
        double* K_val = K.valuePtr();

        // residual r = b - K(A) A and, if requested, its jacobian with the opposite sign in K
        auto assemble = [&](const std::vector<double>& A, std::vector<double>& r, bool jacobian) {
            std::copy(system.b.begin(), system.b.end(), r.begin());
            if (jacobian) {
                std::fill(K_val, K_val + nnz, 0.0);
            }
            for (uint32_t e = 0; e < m; e++) {
                Elem myelem = mesh.data.trianglelist[e];
                const auto& S = system.elem_stiffness[e];
                double a[3] = {A[myelem[0]], A[myelem[1]], A[myelem[2]]};
                double SA[3] = {S[0] * a[0] + S[1] * a[1] + S[2] * a[2],
                                S[1] * a[0] + S[3] * a[1] + S[4] * a[2],
                                S[2] * a[0] + S[4] * a[1] + S[5] * a[2]};
                double nu = elem_nu[e];
                double dnu_dB2 = 0;
                if (elem_spline[e] >= 0) {
                    // A^T S A = measure * B^2
                    double B2 = (a[0] * SA[0] + a[1] * SA[1] + a[2] * SA[2]) / system.elem_measure[e];
                    splines[elem_spline[e]].getNu(B2, nu, dnu_dB2);
                }
                for (uint8_t i = 0; i < 3; i++) {
                    r[myelem[i]] -= nu * SA[i];
                }
                if (jacobian) {
                    // d(nu S A)/dA = nu S + 2 dnu/dB^2 (S A) (S A)^T / measure
                    double w = 2 * dnu_dB2 / system.elem_measure[e];
                    for (uint8_t k = 0; k < 6; k++) {
                        uint8_t i = MagnetostaticSystem::elem_entry_nodes[k][0];
                        uint8_t j = MagnetostaticSystem::elem_entry_nodes[k][1];
                        K_val[system.elem_entries[e][k]] += nu * S[k] + w * SA[i] * SA[j];
                    }
                }
            }
            for (auto node : system.dirichlet_nodes) {
                r[node] = 0;
            }
            if (jacobian) {
                for (int k : fixed_entries) {
                    K_val[k] = 0;
                }
                for (auto node : system.dirichlet_nodes) {
                    K_val[system.col_ptr[node + 1] - 1] = 1;
                }
            }
        };

        auto tstart = std::chrono::high_resolution_clock::now();
        std::vector<double> r(n);
        assemble(A, r, true);

        // the tolerance is relative to the norm of the sources so that it doesn't depend on the units
        double ref_norm = 0;
        for (uint32_t i = 0; i < n; i++) {
            if (!fixed[i]) ref_norm += system.b[i] * system.b[i];
        }
        ref_norm = sqrt(ref_norm);
        if (ref_norm == 0) {
            // no sources, only the dirichlet values drive the solution
            ref_norm = norm(r);
        }
        if (ref_norm == 0) {
            nloginfo("no sources, the solution is zero");
            return A;
        }
        double residual = norm(r) / ref_norm;

        Eigen::SimplicialLDLT<SparseMatrix, Eigen::Upper> ldlt;
        ldlt.analyzePattern(K);

        if (all_linear) {
            nloginfo("all magnetostatic_materials are linear");
        } else {
            nloginfo("nonlinear magnetostatic_materials detected, starting newton solver");
        }
        std::vector<double> r_trial(n);
        std::vector<double> A_trial(n);
        bool converged = false;
        for (uint32_t i = 0; i < 50; i++) {
            ldlt.factorize(K);
            if (ldlt.info() != Eigen::Success) {
                nexit("error: the stiffness matrix could not be factorized");
            }
            Eigen::VectorXd delta = ldlt.solve(Eigen::Map<Eigen::VectorXd>(r.data(), n));

            if (all_linear) {
                // the residual is linear in A, one step solves the system
                for (uint32_t j = 0; j < n; j++) {
                    A[j] += delta[j];
                }
                converged = true;
                break;
            }

            // backtracking line search on the norm of the residual, the jacobian is assembled at every trial
            // so that it is ready for the next iteration
            double step = 1;
            double trial_residual = residual;
            for (uint32_t k = 0; k < 10; k++) {
                for (uint32_t j = 0; j < n; j++) {
                    A_trial[j] = A[j] + step * delta[j];
                }
                assemble(A_trial, r_trial, true);
                trial_residual = norm(r_trial) / ref_norm;
                if (trial_residual < residual) {
                    break;
                }
                step /= 2;
            }
            A.swap(A_trial);
            r.swap(r_trial);
            residual = trial_residual;
            nloginfo("newton iteration %u, relative residual: %.17g, step: %g", i, residual, step);
            if (residual < 1e-10) {
                nloginfo("Converged in %u iterations", i + 1);
                converged = true;
                break;
            }
        }
        if (!converged) {
            nlogwarn("nonlinear solver did not converge, last relative residual: %.17g", residual);
        }
        auto tend = std::chrono::high_resolution_clock::now();
        nloginfo("solver took %f ms", std::chrono::duration_cast<std::chrono::microseconds>(tend - tstart).count() / 1000.0);

        // in axisymmetric problems the kelvin exterior region solves for r' A' = r A, where r' is the radius
        // in the untransformed disk, convert it back to the A_phi of the transformed coordinates
        if (mesh.axisymmetric && mesh.fem_pointlist.size() == n) {
            double R_squared = mesh.radius * mesh.radius;
            for (uint32_t i = mesh.kelvin_first_point; i < n; i++) {
                A[i] *= mesh.fem_pointlist[i].normSquared() / R_squared;
            }
        }

        return A;
    }

    auto MagnetostaticSimulation::getSurroundingRegionBlockIntegralAssets(std::vector<Vector> B, Vector p) {
        // find the polygon that contains p
        Polygon integration_region = mesh.getInnermostPolygon(p);
        Polygon boundary_region;
        std::vector<Polygon> boundary_region_holes;

        // find the region that immediately surrounds the integration region
        std::vector<Polygon> polygons_that_contain_integration_region;
        for (auto polygon : mesh.drawing.polygons) {
            if (polygon.contains(integration_region) && polygon != integration_region) {
                polygons_that_contain_integration_region.push_back(polygon);
            }
        }
        // if there is more than one polygon that contains the integration region then we have to keep
        // removing polygons that aren't contained by any other polygon that contains the integration region
        // until we are left with one polygon
        int count = 0;
        while (polygons_that_contain_integration_region.size() > 1) {
            for (auto polygon : polygons_that_contain_integration_region) {
                bool is_outermost_polygon = true;
                for (auto other_polygon : polygons_that_contain_integration_region) {
                    if (polygon != other_polygon && polygon.contains(other_polygon)) {
                        is_outermost_polygon = false;
                        break;
                    }
                }
                if (is_outermost_polygon) {
                    // remove polygon from polygons_that_contain_integration_region
                    polygons_that_contain_integration_region.erase(
                        std::remove(polygons_that_contain_integration_region.begin(),
                                    polygons_that_contain_integration_region.end(),
                                    polygon),
                        polygons_that_contain_integration_region.end());
                }
            }
            count++;
            if (count > 100) {
                nexit("error: could not find outermost polygon");
            }
        }

        // the boundary region is the polygon that is left
        boundary_region = *polygons_that_contain_integration_region.begin();

        // find the holes in the boundary region (every polygon that is contained by the boundary region but isn't contained by the integration region)
        for (auto polygon : mesh.drawing.polygons) {
            if (boundary_region.contains(polygon) && !integration_region.contains(polygon) && polygon != boundary_region && polygon != integration_region) {
                boundary_region_holes.push_back(polygon);
            }
        }

        auto prop = mesh.drawing.getPolygonProp(integration_region);

        // EGGSHELL WITH LAPLACE PROBLEM

        // COMPUTE FEM WEIGHTS
        auto adjelems_ids = std::vector<std::array<uint32_t, 18>>(mesh.data.numberofpoints);
        auto adjelems_count = std::vector<uint8_t>(mesh.data.numberofpoints, 0);

        nloginfo("computing adjelems_ids");
        for (uint32_t i = 0; i < mesh.data.numberoftriangles; i++) {
            for (uint8_t j = 0; j < 3; j++) {
                uint32_t myid = mesh.data.trianglelist[i][j];
                adjelems_ids[myid][adjelems_count[myid]++] = i;
            }
        }

        nloginfo("computing elemadjelems_ids");
        struct Edge {
            uint8_t count = 0;
            uint32_t elems[2];
        };
        uint64_t edge_count = (mesh.data.numberofpoints - 2) * 3;
        std::unordered_map<uint64_t, Edge> edge_to_triangles(edge_count);
        for (uint32_t i = 0; i < mesh.data.numberoftriangles; i++) {
            for (uint8_t j = 0; j < 3; j++) {
                uint32_t v1 = mesh.data.trianglelist[i][j];
                uint32_t v2 = mesh.data.trianglelist[i][(j + 1) % 3];
                uint64_t edge = v1 < v2 ? (uint64_t)v1 << 32 | v2 : (uint64_t)v2 << 32 | v1;
                edge_to_triangles[edge].elems[edge_to_triangles[edge].count++] = i;
            }
        }

        nloginfo("computing field error");
        // compute field error for each element
        std::vector<double> elem_field_errors = std::vector<double>(mesh.data.numberoftriangles, 0);
        // std::vector<double> elem_field_weights = std::vector<double>(mesh.data.numberoftriangles, 0);
        for (uint32_t i = 0; i < mesh.data.numberoftriangles; i++) {
            double err = (1. / 3.) * B[i].norm();
            Elem myelem = mesh.data.trianglelist[i];
            // make sure the vertices are in counterclockwise order
            if (Vector::doubleOrientedArea(mesh.data.pointlist[myelem[0]],
                                           mesh.data.pointlist[myelem[1]],
                                           mesh.data.pointlist[myelem[2]]) < 0) {
                std::swap(myelem[0], myelem[1]);
            }
            for (uint8_t j = 0; j < 3; j++) {
                uint32_t id1 = myelem[j];
                uint32_t id2 = myelem[(j + 1) % 3];
                Vector v1 = mesh.data.pointlist[id1];
                Vector v2 = mesh.data.pointlist[id2];
                double len = Vector::distance(v1, v2);
                Vector n = (v2 - v1).normal().normalize();
                uint64_t edge = id1 < id2 ? (uint64_t)id1 << 32 | id2 : (uint64_t)id2 << 32 | id1;
                uint32_t adjid1 = edge_to_triangles[edge].elems[0];
                uint32_t adjid2 = edge_to_triangles[edge].elems[1];
                Vector Bp = B[adjid1 == i ? adjid2 : adjid1];
                err += (n ^ (Bp - B[i])) * len;
            }
            elem_field_errors[i] = err;
        }

        nloginfo("computing median field error");
        // compute min and max field error
        double min_err = std::numeric_limits<double>::max();
        double max_err = std::numeric_limits<double>::min();
        for (uint32_t i = 0; i < mesh.data.numberoftriangles; i++) {
            min_err = std::min(min_err, elem_field_errors[i]);
            max_err = std::max(max_err, elem_field_errors[i]);
        }

        nloginfo("find position of vertices");
        std::vector<bool> vertex_inside_integration_region(mesh.data.numberofpoints);
        std::vector<bool> vertex_inside_integration_region_with_boundary(mesh.data.numberofpoints);
        std::vector<bool> vertex_inside_boundary_region(mesh.data.numberofpoints);
        std::vector<bool> vertex_inside_boundary_region_hole(mesh.data.numberofpoints);

        for (uint32_t i = 0; i < mesh.data.numberofpoints; i++) {
            Vector mypoint = mesh.data.pointlist[i];
            bool inside_integration_region_with_boundary = integration_region.contains(mypoint, true, mesh.epsilon);
            bool inside_integration_region = integration_region.contains(mypoint, false, mesh.epsilon);
            bool inside_boundary_region = boundary_region.contains(mypoint, true, mesh.epsilon);
            bool inside_boundary_region_hole = false;
            for (auto& hole : boundary_region_holes) {
                if (hole.contains(mypoint, false, mesh.epsilon)) {
                    inside_boundary_region_hole = true;
                    break;
                }
            }

            vertex_inside_integration_region[i] = inside_integration_region;
            vertex_inside_integration_region_with_boundary[i] = inside_integration_region_with_boundary;
            vertex_inside_boundary_region[i] = inside_boundary_region;
            vertex_inside_boundary_region_hole[i] = inside_boundary_region_hole;
        }

        return SurroundingRegionBlockIntegralAssets{
            adjelems_ids,
            adjelems_count, 
            elem_field_errors, 
            min_err, 
            max_err, 
            vertex_inside_integration_region,
            vertex_inside_integration_region_with_boundary,
            vertex_inside_boundary_region,
            vertex_inside_boundary_region_hole
        };
    }

    Vector MagnetostaticSimulation::computeForceIntegrals(std::vector<Vector> B, SurroundingRegionBlockIntegralAssets assets, Vector p) {
        if (mesh.axisymmetric) {
            nexit("error: forces and torques are not implemented for axisymmetric problems yet");
        }
        auto start = std::chrono::high_resolution_clock::now();
        
        auto [
            adjelems_ids, 
            adjelems_count, 
            elem_field_errors, 
            min_err, 
            max_err, 
            vertex_inside_integration_region,
            vertex_inside_integration_region_with_boundary,
            vertex_inside_boundary_region,
            vertex_inside_boundary_region_hole
        ] = assets;

        auto coo = MatCOOSymmetric<double>(mesh.data.numberofpoints);
        auto b = std::vector<double>(mesh.data.numberofpoints);
        auto b_dirichlet_mask = std::vector<bool>(mesh.data.numberofpoints, false);
        // since the stiffness matrix is symmetric, this function only computes the upper triangular part

        // square root of 2 / 2
        Vector myS = Vector(1, 0).normalize();

        nloginfo("building fem matrix");
        // surface integral
        for (uint32_t i = 0; i < mesh.data.numberofpoints; i++) {
            for (uint8_t j = 0; j < adjelems_count[i]; j++) {
                uint32_t v1, v2, v3;
                v1 = i;
                Elem myelem = mesh.data.trianglelist[adjelems_ids[i][j]];
                Vector myB = B[adjelems_ids[i][j]];
                if (i == mesh.data.trianglelist[adjelems_ids[i][j]][0]) {
                    v2 = myelem[1];
                    v3 = myelem[2];
                } else if (i == myelem[1]) {
                    v2 = myelem[2];
                    v3 = myelem[0];
                } else if (i == myelem[2]) {
                    v2 = myelem[0];
                    v3 = myelem[1];
                } else {
                    nexit("error: vertex not found in element");
                }

                double oriented_area = mesh.data.getDoubleOrientedArea(v1, v2, v3);
                
                if (oriented_area < 0) {
                    std::swap(v2, v3);
                }

                double area = mesh.data.getDoubleOrientedArea(v1, v2, v3);

                // elements are only added if they are in the upper triangle because the matrix is symmetric and this saves half the memory
                double b1 = (mesh.data.pointlist[v2].y - mesh.data.pointlist[v3].y) / area;
                double b2 = (mesh.data.pointlist[v3].y - mesh.data.pointlist[v1].y) / area;
                double b3 = (mesh.data.pointlist[v1].y - mesh.data.pointlist[v2].y) / area;
                double c1 = (mesh.data.pointlist[v3].x - mesh.data.pointlist[v2].x) / area;
                double c2 = (mesh.data.pointlist[v1].x - mesh.data.pointlist[v3].x) / area;
                double c3 = (mesh.data.pointlist[v2].x - mesh.data.pointlist[v1].x) / area;

                double K1 = - 0.5 * myS.x * myB.x * myB.x + 0.5 * myS.x * myB.y * myB.y - myS.y * myB.x * myB.y;
                double K2 = - 0.5 * myS.y * myB.y * myB.y + 0.5 * myS.y * myB.x * myB.x - myS.x * myB.x * myB.y;

                double err = elem_field_errors[adjelems_ids[i][j]];
                
                double Wi = limit(map(sqrt(fabs(err)), min_err, max_err, 1, 1000), 1, 1000);
                // elem_field_weights[adjelems_ids[i][j]] = Wi;
                
                if (v1 >= i) coo(i, v1) += area * Wi * (K1 * b1 + K2 * c1) * (K1 * b1 + K2 * c1);
                if (v2 >= i) coo(i, v2) += area * Wi * (K1 * b1 + K2 * c1) * (K1 * b2 + K2 * c2);
                if (v3 >= i) coo(i, v3) += area * Wi * (K1 * b1 + K2 * c1) * (K1 * b3 + K2 * c3);

                // if (v1 >= i) coo(i, v1) += area * 0.5 * Wi * Wi * (b1 * b1 + c1 * c1);
                // if (v2 >= i) coo(i, v2) += area * 0.5 * Wi * Wi * (b1 * b2 + c1 * c2);
                // if (v3 >= i) coo(i, v3) += area * 0.5 * Wi * Wi * (b1 * b3 + c1 * c3);
            }
        }

        nloginfo("dirichlet boundary conditions");
        // std::vector<double> test(mesh.data.numberofpoints, 0);
        // we have to set a 1 dirichlet boundary condition for all the vertices inside the integration region

        uint32_t val = 0;
        for (auto& [id, value] : coo.elems) {
            uint32_t col = id >> 32;
            uint32_t row = id & 0xFFFFFFFF;

            bool col_in_zero_region = !vertex_inside_boundary_region[col] || vertex_inside_boundary_region_hole[col];
            bool col_in_one_region = vertex_inside_integration_region_with_boundary[col];

            double old_value = value;

            if (col_in_zero_region) {
                if (row == col) {
                    value = 1;
                    b[row] = 0;
                } else {
                    value = 0;
                }
            } else if (col_in_one_region) {
                if (row == col) {
                    value = 1;
                    b[row] = 1;
                } else {
                    value = 0;
                    b[row] -= old_value;
                }
            }
            val++;
        }

        MatCSRSymmetric Ai(coo);
        // Ai.write_to_file("Ad.txt");
        auto g = std::vector<double>(mesh.data.numberofpoints);

        // the error for this function varies a lot, so I just set the gradient to do at least 100 iterations and eventually stop at 1e-100
        preconditionedSSORConjugateGradientSolver(Ai, b, g, 1.5, 1e-100, 100);

        // compute grad(g)
        std::vector<Vector> nabla_g(mesh.data.numberoftriangles, {0, 0});
        nabla_g = mesh.computeGrad(g);

        std::vector<double> nabla_g_norm(mesh.data.numberoftriangles, 0);
        for (uint32_t i = 0; i < mesh.data.numberoftriangles; i++) {
            nabla_g_norm[i] = nabla_g[i].norm();
        }

        // integrate force
        Vector force = {0, 0};

        for (uint32_t i = 0; i < mesh.data.numberoftriangles; i++) {
            bool inside = true;
            for (uint32_t j = 0; j < 3; j++) {
                uint32_t v = mesh.data.trianglelist[i][j];
                if (vertex_inside_integration_region[v] || !vertex_inside_boundary_region[v] || vertex_inside_boundary_region_hole[v]) {
                    inside = false;
                    break;
                }
            }

            if (inside) {
                uint32_t v1 = mesh.data.trianglelist[i][0];
                uint32_t v2 = mesh.data.trianglelist[i][1];
                uint32_t v3 = mesh.data.trianglelist[i][2];

                Vector p1 = mesh.data.pointlist[v1];
                Vector p2 = mesh.data.pointlist[v2];
                Vector p3 = mesh.data.pointlist[v3];

                Vector my_nabla_g = nabla_g[i];
                Vector my_B = B[i];

                double area = Vector::area(p1, p2, p3);

                force += (my_nabla_g * 0.5 * my_B.normSquared() - my_B * (my_nabla_g * my_B)) * area;
            }
        }
        force *= 1 / (4 * PI * 1e-7);
        force *= depth;

        auto end = std::chrono::high_resolution_clock::now();
        nloginfo("Time elapsed: %f ms", std::chrono::duration<double, std::milli>(end - start).count());
        return force;
    }

    double MagnetostaticSimulation::computeTorqueIntegral(std::vector<Vector> B, SurroundingRegionBlockIntegralAssets assets, Vector p, Vector center) {
        if (mesh.axisymmetric) {
            nexit("error: forces and torques are not implemented for axisymmetric problems yet");
        }
        auto start = std::chrono::high_resolution_clock::now();
        
        auto [
            adjelems_ids, 
            adjelems_count, 
            elem_field_errors, 
            min_err, 
            max_err, 
            vertex_inside_integration_region,
            vertex_inside_integration_region_with_boundary,
            vertex_inside_boundary_region,
            vertex_inside_boundary_region_hole
        ] = assets;

        auto coo = MatCOOSymmetric<double>(mesh.data.numberofpoints);
        auto b = std::vector<double>(mesh.data.numberofpoints);
        auto b_dirichlet_mask = std::vector<bool>(mesh.data.numberofpoints, false);
        // since the stiffness matrix is symmetric, this function only computes the upper triangular part

        nloginfo("building fem matrix");
        // surface integral
        for (uint32_t i = 0; i < mesh.data.numberofpoints; i++) {
            for (uint8_t j = 0; j < adjelems_count[i]; j++) {
                uint32_t v1, v2, v3;
                v1 = i;
                Elem myelem = mesh.data.trianglelist[adjelems_ids[i][j]];
                Vector myB = B[adjelems_ids[i][j]];
                if (i == mesh.data.trianglelist[adjelems_ids[i][j]][0]) {
                    v2 = myelem[1];
                    v3 = myelem[2];
                } else if (i == myelem[1]) {
                    v2 = myelem[2];
                    v3 = myelem[0];
                } else if (i == myelem[2]) {
                    v2 = myelem[0];
                    v3 = myelem[1];
                } else {
                    nexit("error: vertex not found in element");
                }

                double oriented_area = mesh.data.getDoubleOrientedArea(v1, v2, v3);
                
                if (oriented_area < 0) {
                    std::swap(v2, v3);
                }

                double area = mesh.data.getDoubleOrientedArea(v1, v2, v3);

                // elements are only added if they are in the upper triangle because the matrix is symmetric and this saves half the memory
                double b1 = (mesh.data.pointlist[v2].y - mesh.data.pointlist[v3].y) / area;
                double b2 = (mesh.data.pointlist[v3].y - mesh.data.pointlist[v1].y) / area;
                double b3 = (mesh.data.pointlist[v1].y - mesh.data.pointlist[v2].y) / area;
                double c1 = (mesh.data.pointlist[v3].x - mesh.data.pointlist[v2].x) / area;
                double c2 = (mesh.data.pointlist[v1].x - mesh.data.pointlist[v3].x) / area;
                double c3 = (mesh.data.pointlist[v2].x - mesh.data.pointlist[v1].x) / area;

                Vector barycenter = (mesh.data.pointlist[v1] + mesh.data.pointlist[v2] + mesh.data.pointlist[v3]) / 3;
                Vector r = barycenter - center;

                double K1 = - r.y * 0.5 * myB.normSquared() - myB.x * (r ^ myB);
                double K2 = r.x * 0.5 * myB.normSquared() - myB.y * (r ^ myB);

                double err = elem_field_errors[adjelems_ids[i][j]];
                
                double Wi = limit(map(sqrt(fabs(err)), min_err, max_err, 1, 1000), 1, 1000);
                // elem_field_weights[adjelems_ids[i][j]] = Wi;
                
                if (v1 >= i) coo(i, v1) += area * Wi * (K1 * b1 + K2 * c1) * (K1 * b1 + K2 * c1);
                if (v2 >= i) coo(i, v2) += area * Wi * (K1 * b1 + K2 * c1) * (K1 * b2 + K2 * c2);
                if (v3 >= i) coo(i, v3) += area * Wi * (K1 * b1 + K2 * c1) * (K1 * b3 + K2 * c3);

                // if (v1 >= i) coo(i, v1) += area * 0.5 * Wi * Wi * (b1 * b1 + c1 * c1);
                // if (v2 >= i) coo(i, v2) += area * 0.5 * Wi * Wi * (b1 * b2 + c1 * c2);
                // if (v3 >= i) coo(i, v3) += area * 0.5 * Wi * Wi * (b1 * b3 + c1 * c3);
            }
        }

        nloginfo("dirichlet boundary conditions");
        // std::vector<double> test(mesh.data.numberofpoints, 0);
        // we have to set a 1 dirichlet boundary condition for all the vertices inside the integration region

        uint32_t val = 0;
        for (auto& [id, value] : coo.elems) {
            uint32_t col = id >> 32;
            uint32_t row = id & 0xFFFFFFFF;

            bool col_in_zero_region = !vertex_inside_boundary_region[col] || vertex_inside_boundary_region_hole[col];
            bool col_in_one_region = vertex_inside_integration_region_with_boundary[col];

            double old_value = value;

            if (col_in_zero_region) {
                if (row == col) {
                    value = 1;
                    b[row] = 0;
                } else {
                    value = 0;
                }
            } else if (col_in_one_region) {
                if (row == col) {
                    value = 1;
                    b[row] = 1;
                } else {
                    value = 0;
                    b[row] -= old_value;
                }
            }
            val++;
        }

        MatCSRSymmetric Ai(coo);
        // Ai.write_to_file("Ad.txt");
        auto g = std::vector<double>(mesh.data.numberofpoints);

        // the error for this function varies a lot, so I just set the gradient to do at least 100 iterations and eventually stop at 1e-100
        preconditionedSSORConjugateGradientSolver(Ai, b, g, 1.5, 1e-100, 100);

        // compute grad(g)
        std::vector<Vector> nabla_g(mesh.data.numberoftriangles, {0, 0});
        nabla_g = mesh.computeGrad(g);

        std::vector<double> nabla_g_norm(mesh.data.numberoftriangles, 0);
        for (uint32_t i = 0; i < mesh.data.numberoftriangles; i++) {
            nabla_g_norm[i] = nabla_g[i].norm();
        }

        // integrate force
        double torque = 0;

        for (uint32_t i = 0; i < mesh.data.numberoftriangles; i++) {
            bool inside = true;
            for (uint32_t j = 0; j < 3; j++) {
                uint32_t v = mesh.data.trianglelist[i][j];
                if (vertex_inside_integration_region[v] || !vertex_inside_boundary_region[v] || vertex_inside_boundary_region_hole[v]) {
                    inside = false;
                    break;
                }
            }

            if (inside) {
                uint32_t v1 = mesh.data.trianglelist[i][0];
                uint32_t v2 = mesh.data.trianglelist[i][1];
                uint32_t v3 = mesh.data.trianglelist[i][2];

                Vector p1 = mesh.data.pointlist[v1];
                Vector p2 = mesh.data.pointlist[v2];
                Vector p3 = mesh.data.pointlist[v3];

                Vector my_nabla_g = nabla_g[i];
                Vector my_B = B[i];

                double area = Vector::area(p1, p2, p3);

                Vector barycenter = (p1 + p2 + p3) / 3;
                Vector r = barycenter - center;

                torque += ((r ^ my_nabla_g) * my_B.normSquared() * 0.5 - (r ^ my_B) * (my_B * my_nabla_g)) * area;
                // force += (my_nabla_g * 0.5 * my_B.normSquared() - my_B * (my_nabla_g * my_B)) * area;
            }
        }
        torque *= 1 / (4 * PI * 1e-7);
        torque *= depth;

        auto end = std::chrono::high_resolution_clock::now();
        nloginfo("Time elapsed: %f ms", std::chrono::duration<double, std::milli>(end - start).count());
        return torque;
    }

    Vector MagnetostaticSimulation::computeForceIntegrals(std::vector<Vector> B, Vector p) {
        auto assets = getSurroundingRegionBlockIntegralAssets(B, p);
        return computeForceIntegrals(B, assets, p);
    }

    double MagnetostaticSimulation::computeTorqueIntegral(std::vector<Vector> B, Vector p, Vector center) {
        auto assets = getSurroundingRegionBlockIntegralAssets(B, p);
        return computeTorqueIntegral(B, assets, p, center);
    }

    StressTensor MagnetostaticSimulation::computeStressIntegral(std::vector<Vector> B, Vector p, Vector center) {
        auto assets = getSurroundingRegionBlockIntegralAssets(B, p);
        Vector force = computeForceIntegrals(B, assets, p);
        double torque = computeTorqueIntegral(B, assets, p, center);
        return {force, torque};
    }
}