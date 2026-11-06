#include "mesh.hpp"

namespace nikfemm {
    MagnetostaticMesh::MagnetostaticMesh() {
        default_prop = {0, {0, 0}, magnetostatic_materials::air, {}};
    }

    MagnetostaticSystem MagnetostaticMesh::getFemSystem() {
        MagnetostaticSystem system;
        uint32_t n = data.numberofpoints;
        uint32_t m = data.numberoftriangles;
        system.n = n;
        system.b = std::vector<double>(n, 0);
        system.elem_stiffness.resize(m);
        system.elem_entries.resize(m);
        system.elem_measure.resize(m);

        auto start = std::chrono::high_resolution_clock::now();

        // the kelvin exterior region is assembled in its untransformed coordinates
        bool has_kelvin = fem_pointlist.size() == (size_t)data.numberofpoints;
        const Vector* points = has_kelvin ? fem_pointlist.data() : data.pointlist;

        // 7 point gauss rule (exact up to degree 5) for the radius weighted integrals of axisymmetric problems
        // barycentric coordinates and weight of each point
        static const double gauss_points[7][4] = {
            {1. / 3., 1. / 3., 1. / 3., 0.225},
            {0.059715871789770, 0.470142064105115, 0.470142064105115, 0.132394152788506},
            {0.470142064105115, 0.059715871789770, 0.470142064105115, 0.132394152788506},
            {0.470142064105115, 0.470142064105115, 0.059715871789770, 0.132394152788506},
            {0.797426985353087, 0.101286507323456, 0.101286507323456, 0.125939180544827},
            {0.101286507323456, 0.797426985353087, 0.101286507323456, 0.125939180544827},
            {0.101286507323456, 0.101286507323456, 0.797426985353087, 0.125939180544827}
        };

        // surface integral, element by element
        for (uint32_t e = 0; e < m; e++) {
            Elem myelem = data.trianglelist[e];
            Vector p[3] = {points[myelem[0]], points[myelem[1]], points[myelem[2]]};
            double double_area = (p[1] - p[0]) ^ (p[2] - p[0]);
            double area = fabs(double_area) / 2;

            // gradients of the shape functions times twice the signed area
            double b[3] = {p[1].y - p[2].y, p[2].y - p[0].y, p[0].y - p[1].y};
            double c[3] = {p[2].x - p[1].x, p[0].x - p[2].x, p[1].x - p[0].x};
            double J = drawing.getRegionPtrFromId(data.triangleattributelist[e])->J;

            if (!axisymmetric) {
                // the sign of the area doesn't matter in the products
                for (uint8_t k = 0; k < 6; k++) {
                    uint8_t i = MagnetostaticSystem::elem_entry_nodes[k][0];
                    uint8_t j = MagnetostaticSystem::elem_entry_nodes[k][1];
                    system.elem_stiffness[e][k] = (b[i] * b[j] + c[i] * c[j]) / (4 * area);
                }
                system.elem_measure[e] = area;

                // set the b vector
                for (uint8_t i = 0; i < 3; i++) {
                    system.b[myelem[i]] += J * area / 3;
                }
            } else {
                // the unknown is A_phi with linear shape functions, x is the radius and y the axial coordinate
                // B_r = -dA/dz and B_z = dA/dr + A/r, the energy is the integral of nu B^2 r / 2 (per radian)
                double dNdr[3], dNdz[3];
                for (uint8_t i = 0; i < 3; i++) {
                    dNdr[i] = b[i] / double_area;
                    dNdz[i] = c[i] / double_area;
                }
                // the kelvin transformation of an axisymmetric exterior region needs a reluctivity
                // proportional to the square of the distance from the center of the disk
                bool kelvin = has_kelvin && e >= kelvin_first_triangle;
                system.elem_stiffness[e].fill(0);
                for (auto& g : gauss_points) {
                    double r = g[0] * p[0].x + g[1] * p[1].x + g[2] * p[2].x;
                    double z = g[0] * p[0].y + g[1] * p[1].y + g[2] * p[2].y;
                    double w = g[3] * area * (kelvin ? (r * r + z * z) / (radius * radius) : 1);
                    for (uint8_t k = 0; k < 6; k++) {
                        uint8_t i = MagnetostaticSystem::elem_entry_nodes[k][0];
                        uint8_t j = MagnetostaticSystem::elem_entry_nodes[k][1];
                        // (dN_i/dr + N_i/r)(dN_j/dr + N_j/r) r + dN_i/dz dN_j/dz r
                        system.elem_stiffness[e][k] += w * (r * (dNdr[i] * dNdr[j] + dNdz[i] * dNdz[j]) +
                                                            g[i] * dNdr[j] + g[j] * dNdr[i] + g[i] * g[j] / r);
                    }
                }
                // integral of r over the element
                system.elem_measure[e] = area * (p[0].x + p[1].x + p[2].x) / 3;

                // the integral of N_i r over the element
                for (uint8_t i = 0; i < 3; i++) {
                    system.b[myelem[i]] += J * area * (p[i].x + p[0].x + p[1].x + p[2].x) / 12;
                }
            }
        }

        // sparsity pattern of the upper triangle: for every column, the rows of its entries
        // every column gets its diagonal even if no element touches the node
        std::vector<int> col_count(n + 1, 0);
        for (uint32_t e = 0; e < m; e++) {
            Elem myelem = data.trianglelist[e];
            for (uint8_t k = 0; k < 6; k++) {
                int col = std::max(myelem[MagnetostaticSystem::elem_entry_nodes[k][0]], myelem[MagnetostaticSystem::elem_entry_nodes[k][1]]);
                col_count[col + 1]++;
            }
        }
        for (uint32_t i = 0; i < n; i++) {
            col_count[i + 1]++;
            col_count[i + 1] += col_count[i];
        }
        std::vector<int> candidate_rows(col_count[n]);
        std::vector<int> cursor(col_count.begin(), col_count.end() - 1);
        for (uint32_t i = 0; i < n; i++) {
            candidate_rows[cursor[i]++] = i;
        }
        for (uint32_t e = 0; e < m; e++) {
            Elem myelem = data.trianglelist[e];
            for (uint8_t k = 0; k < 6; k++) {
                int v1 = myelem[MagnetostaticSystem::elem_entry_nodes[k][0]];
                int v2 = myelem[MagnetostaticSystem::elem_entry_nodes[k][1]];
                candidate_rows[cursor[std::max(v1, v2)]++] = std::min(v1, v2);
            }
        }
        system.col_ptr = std::vector<int>(n + 1, 0);
        system.row_ind.reserve(col_count[n] / 2);
        for (uint32_t j = 0; j < n; j++) {
            auto first = candidate_rows.begin() + col_count[j];
            auto last = candidate_rows.begin() + col_count[j + 1];
            std::sort(first, last);
            last = std::unique(first, last);
            system.row_ind.insert(system.row_ind.end(), first, last);
            system.col_ptr[j + 1] = system.row_ind.size();
        }

        // position of every element entry in the sparse matrix
        for (uint32_t e = 0; e < m; e++) {
            Elem myelem = data.trianglelist[e];
            for (uint8_t k = 0; k < 6; k++) {
                int v1 = myelem[MagnetostaticSystem::elem_entry_nodes[k][0]];
                int v2 = myelem[MagnetostaticSystem::elem_entry_nodes[k][1]];
                int row = std::min(v1, v2);
                int col = std::max(v1, v2);
                auto first = system.row_ind.begin() + system.col_ptr[col];
                auto last = system.row_ind.begin() + system.col_ptr[col + 1];
                system.elem_entries[e][k] = std::lower_bound(first, last, row) - system.row_ind.begin();
            }
        }

        // line integral
        // in this section we find all the edges where there is a surface current (i.e. the edges where the M vector changes from element to element)
        // to do this first of all we find all the segments in the drawing that are subject to a surface current and their outward normal
        struct SurfaceCurrentSegment {
            Vector p1, p2; // the two points that define the segment
            double Jm; // surface current density on the segment (M x n) (pointing normal to the simulation plane)
        };

        // first we have to find the polygons that are magnets
        // if the polygon contains a region that is a magnet and it does not contain any other polygon that contains the same region then it is a magnet
        std::map<uint32_t, std::vector<Polygon>> polygons_that_contain_magnet_region;
        for (auto region :drawing.regions) {
            if (drawing.region_map[region.second].M != Vector(0, 0)) {
                for (auto polygon : drawing.polygons) {
                    if (polygon.contains(region.first)) {
                        polygons_that_contain_magnet_region[region.second].push_back(polygon);
                    }
                }
            }
        }

        struct Magnet {
            Polygon polygon;
            Vector M;
            std::vector<SurfaceCurrentSegment> segments;
        };

        std::vector<Magnet> magnets;
        for (auto region_poly_array : polygons_that_contain_magnet_region) {
            if (region_poly_array.second.size() == 1) {
                // this is a magnet
                magnets.push_back({region_poly_array.second[0], drawing.region_map[region_poly_array.first].M, {}});
            } else {
                // this array contains a magnet and other polygons that contain the polygon that is a magnet
                // we need to find the polygon that does not contain any other polygon that contains the same region
                for (auto polygon : region_poly_array.second) {
                    bool is_magnet = true;
                    for (auto other_polygon : region_poly_array.second) {
                        if (polygon != other_polygon && polygon.contains(other_polygon)) {
                            is_magnet = false;
                            break;
                        }
                    }
                    if (is_magnet) {
                        magnets.push_back({polygon, drawing.region_map[region_poly_array.first].M, {}});
                    }
                }
            }
        }

        // now we have to find the segments that are subject to a surface current and their outward normal
        for (auto &magnet : magnets) {
            for (uint32_t i = 0; i < magnet.polygon.points.size(); i++) {
                uint32_t j = (i + 1) % magnet.polygon.points.size();
                Vector p1 = magnet.polygon.points[i];
                Vector p2 = magnet.polygon.points[j];
                Vector n = (p2 - p1).normal().normalize();
                // we need the outward normal
                Vector test = Vector::midPoint(p1, p2) + (n * epsilon * 0.1);  // 0.1 just to make sure that we get no false negatives
                if (magnet.polygon.contains(test)) {
                    n = n * -1;
                }
                // draw arrow from the middle of p1 and p2 in the direction of n
                Vector mid = Vector::midPoint(p1, p2);
                double Jm = magnet.M ^ n;
                magnet.segments.push_back({p1, p2, Jm});
            }
        }

        // now we have to find the edges that are subject to a surface current
        // i.e. the edges that lie on a segment that is subject to a surface current
        // these can only be subsegments of the drawing segments, which triangle outputs as a unique list
        double tolerance = epsilon * 0.1;
        // a point within the tolerance of a segment is inside the bounding box of the segment grown by the tolerance
        auto inSegmentBox = [tolerance](Vector p, const SurfaceCurrentSegment& s) {
            return p.x >= std::min(s.p1.x, s.p2.x) - tolerance && p.x <= std::max(s.p1.x, s.p2.x) + tolerance &&
                   p.y >= std::min(s.p1.y, s.p2.y) - tolerance && p.y <= std::max(s.p1.y, s.p2.y) + tolerance;
        };
        for (int s = 0; s < data.numberofsegments; s++) {
            uint32_t v1 = data.segmentlist[2 * s];
            uint32_t v2 = data.segmentlist[2 * s + 1];
            Vector p1 = data.pointlist[v1];
            Vector p2 = data.pointlist[v2];
            for (auto& magnet : magnets) {
                for (auto& segment : magnet.segments) {
                    if (!inSegmentBox(p1, segment) || !inSegmentBox(p2, segment)) {
                        continue;
                    }
                    double dist1 = Segment::pointSegmentDistance(p1, segment.p1, segment.p2);
                    double dist2 = Segment::pointSegmentDistance(p2, segment.p1, segment.p2);
                    if (dist1 < tolerance && dist2 < tolerance) {
                        // this edge is subject to a surface current
                        // the integral of a linear shape function over the edge is length / 2
                        double length = Vector::distance(p1, p2);
                        if (!axisymmetric) {
                            system.b[v1] += length * segment.Jm * 0.5;
                            system.b[v2] += length * segment.Jm * 0.5;
                        } else {
                            // the out of plane direction of the drawing is -phi, and the line integral is weighted by r
                            double K = -segment.Jm;
                            system.b[v1] += K * length * (2 * p1.x + p2.x) / 6;
                            system.b[v2] += K * length * (p1.x + 2 * p2.x) / 6;
                        }
                    }
                }
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        nloginfo("FEM matrix construction took %d ms", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
        

        return system;
    }

    void MagnetostaticMesh::addDirichletInfiniteBoundaryConditions(MagnetostaticSystem& system) {
        // in axisymmetric problems A_phi is zero on the symmetry axis, which also contains the point at infinity
        if (axisymmetric) {
            const Vector* points = fem_pointlist.size() == (size_t)data.numberofpoints ? fem_pointlist.data() : data.pointlist;
            for (uint32_t i = 0; i < (uint32_t)data.numberofpoints; i++) {
                if (fabs(points[i].x) <= 1e-12 * radius) {
                    system.addDirichletBoundaryCondition(i, 0);
                }
            }
            return;
        }

        // with the kelvin exterior region the potential is zero at infinity, i.e. at the center of the kelvin disk
        if (kelvin_center_point >= 0) {
            system.addDirichletBoundaryCondition(kelvin_center_point, 0);
            return;
        }

        // find three furthest points from the center
        uint32_t p1 = 0;
        uint32_t p2 = 0;
        uint32_t p3 = 0;
        double d1 = 0;
        double d2 = 0;
        double d3 = 0;
        for (uint32_t i = 0; i < data.numberofpoints; i++) {
            double dist = Vector::distance(data.pointlist[i], Vector(0, 0));
            if (dist > d1) {
                d3 = d2;
                p3 = p2;
                d2 = d1;
                p2 = p1;
                d1 = dist;
                p1 = i;
            } else if (dist > d2) {
                d3 = d2;
                p3 = p2;
                d2 = dist;
                p2 = i;
            } else if (dist > d3) {
                d3 = dist;
                p3 = i;
            }
        }

        // make the magnetic vector potential zero on the three points
        system.addDirichletBoundaryCondition(p1, 0);
        system.addDirichletBoundaryCondition(p2, 0);
        system.addDirichletBoundaryCondition(p3, 0);
    }

    std::vector<Vector> MagnetostaticMesh::computeCurl(std::vector<double>& in) const {
        if (!axisymmetric) {
            return Mesh<MagnetostaticProp>::computeCurl(in);
        }
        // B_r = -dA/dz and B_z = dA/dr + A/r at the barycenter of each element
        std::vector<Vector> out = computeGrad(in);
        for (uint32_t i = 0; i < (uint32_t)data.numberoftriangles; i++) {
            Elem myelem = data.trianglelist[i];
            double r = (data.pointlist[myelem[0]].x + data.pointlist[myelem[1]].x + data.pointlist[myelem[2]].x) / 3;
            double A = (in[myelem[0]] + in[myelem[1]] + in[myelem[2]]) / 3;
            out[i] = Vector(-out[i].y, out[i].x + A / r);
        }
        return out;
    }

    void MagnetostaticMesh::refineMeshAroundMagnets() {

        // add magnets refining points
        // first we have to find the polygons that are magnets
        // if the polygon contains a region that is a magnet and it does not contain any other polygon that contains the same region then it is a magnet
        std::map<uint32_t, std::vector<Polygon>> polygons_that_contain_magnet_region;
        for (auto region :drawing.regions) {
            if (drawing.region_map[region.second].M != Vector(0, 0)) {
                for (auto polygon : drawing.polygons) {
                    if (polygon.contains(region.first)) {
                        polygons_that_contain_magnet_region[region.second].push_back(polygon);
                    }
                }
            }
        }
        std::vector<Polygon> magnets;
        for (auto region_poly_array : polygons_that_contain_magnet_region) {
            if (region_poly_array.second.size() == 1) {
                // this is a magnet
                magnets.push_back(region_poly_array.second[0]);
            } else {
                // this array contains a magnet and other polygons that contain the polygon that is a magnet
                // we need to find the polygon that does not contain any other polygon that contains the same region
                for (auto polygon : region_poly_array.second) {
                    bool is_magnet = true;
                    for (auto other_polygon : region_poly_array.second) {
                        if (polygon != other_polygon && polygon.contains(other_polygon)) {
                            is_magnet = false;
                            break;
                        }
                    }
                    if (is_magnet) {
                        magnets.push_back(polygon);
                    }
                }
            }
        }
        nloginfo("Found %lu magnets", magnets.size());
        for (auto polygon : magnets) {
            for (uint32_t i = 0; i < polygon.points.size(); i++) {
                Vector p1 = polygon.points[i];
                Vector p2 = polygon.points[(i + 1) % polygon.points.size()];
                
                double multiplier = 10;
                double epsilon = drawing.getCurrentEpsilon();
                uint32_t n_points = Vector::distance(p1, p2) / (epsilon / multiplier);
                // multiply by 1.1 to make sure that the points are far enough to not trigger conformity checks
                // Vector normal = Vector::normal(p1, p2).versor() * (drawing.epsilon / multiplier) * 1.1;
                Vector normal = (p2 - p1).normal().normalize() * (epsilon / multiplier) * 1.1;

                for (uint32_t j = 0; j < n_points; j++) {
                    Vector mypoints[2];
                    mypoints[0] = Vector::lerp(p1, p2, (double)j / n_points) + normal;
                    mypoints[1] = Vector::lerp(p1, p2, (double)j / n_points) - normal;

                    bool mypoints_ok[2] = {
                        polygon.contains(mypoints[0]), 
                        polygon.contains(mypoints[1])
                    };

                    // check points at least drawing.epsilon away from each edge of each polygon
                    for (uint8_t k = 0; k < 2; k++) {
                        // skip check for failed points
                        if (mypoints_ok[k]) {
                            // for every polygon
                            for (auto other_polygon : drawing.polygons) {
                                // for every edge of the other polygon
                                for (uint32_t l = 0; l < other_polygon.points.size(); l++) {
                                    // check if the point is at least drawing.epsilon away from the edge
                                    Vector p3 = other_polygon.points[l];
                                    Vector p4 = other_polygon.points[(l + 1) % other_polygon.points.size()];
                                    if (Vector::distance(mypoints[k], p3) < epsilon / multiplier ||
                                        Vector::distance(mypoints[k], p4) < epsilon / multiplier) {
                                        mypoints_ok[k] = false;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    // check points at least drawing.epsilon away from each point in drawing.points
                    for (uint8_t k = 0; k < 2; k++) {
                        // skip check for failed points
                        if (mypoints_ok[k]) {
                            for (auto point : drawing.points) {
                                if (Vector::distance(mypoints[k], point) < epsilon / multiplier) {
                                    mypoints_ok[k] = false;
                                    break;
                                }
                            }
                        }
                    }

                    // add points to drawing.points
                    for (uint8_t k = 0; k < 2; k++) {
                        if (mypoints_ok[k]) {
                            drawing.points.push_back(mypoints[k]);
                        }
                    }
                }
            }
        }
    }
}