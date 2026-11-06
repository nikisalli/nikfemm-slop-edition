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
        system.elem_area.resize(m);

        auto start = std::chrono::high_resolution_clock::now();

        // the kelvin exterior region is assembled in its untransformed coordinates
        const Vector* points = fem_pointlist.size() == (size_t)data.numberofpoints ? fem_pointlist.data() : data.pointlist;

        // surface integral, element by element
        for (uint32_t e = 0; e < m; e++) {
            Elem myelem = data.trianglelist[e];
            Vector p0 = points[myelem[0]];
            Vector p1 = points[myelem[1]];
            Vector p2 = points[myelem[2]];
            double area = fabs((p1 - p0) ^ (p2 - p0)) / 2;

            // gradients of the shape functions times twice the area, the sign doesn't matter in the products
            double b[3] = {p1.y - p2.y, p2.y - p0.y, p0.y - p1.y};
            double c[3] = {p2.x - p1.x, p0.x - p2.x, p1.x - p0.x};
            for (uint8_t k = 0; k < 6; k++) {
                uint8_t i = MagnetostaticSystem::elem_entry_nodes[k][0];
                uint8_t j = MagnetostaticSystem::elem_entry_nodes[k][1];
                system.elem_stiffness[e][k] = (b[i] * b[j] + c[i] * c[j]) / (4 * area);
            }
            system.elem_area[e] = area;

            // set the b vector
            double J = drawing.getRegionPtrFromId(data.triangleattributelist[e])->J;
            for (uint8_t i = 0; i < 3; i++) {
                system.b[myelem[i]] += J * area / 3;
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
                        system.b[v1] += length * segment.Jm * 0.5;
                        system.b[v2] += length * segment.Jm * 0.5;
                    }
                }
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        nloginfo("FEM matrix construction took %d ms", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
        

        return system;
    }

    void MagnetostaticMesh::addDirichletInfiniteBoundaryConditions(MagnetostaticSystem& system) {
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