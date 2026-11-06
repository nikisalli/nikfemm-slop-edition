#ifndef NIK_MAGNETOSTATIC_SIMULATION_H
#define NIK_MAGNETOSTATIC_SIMULATION_H

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cstdint>
#include <unordered_map>

#include "../triangle/triangle.h"

#include "../../src/triangle/util.h"
#include "../constants.hpp"
#include "../geometry/segment.hpp"
#include "../geometry/circle.hpp"
#include "../drawing/drawing.hpp"
#include "../magnetostatic/mesh.hpp"
#include "mesh.hpp"

namespace nikfemm {
    struct SurroundingRegionBlockIntegralAssets {
        std::vector<std::array<uint32_t, 18>> adjelems_ids;
        std::vector<uint8_t> adjelems_count;
        std::vector<double> elem_field_errors;
        double min_err;
        double max_err;
        std::vector<bool> vertex_inside_integration_region;
        std::vector<bool> vertex_inside_integration_region_with_boundary;
        std::vector<bool> vertex_inside_boundary_region;
        std::vector<bool> vertex_inside_boundary_region_hole;
    };

    struct StressTensor {
        Vector Force;
        double Torque;
    };

    class MagnetostaticSimulation {
        public:
            MagnetostaticMesh mesh;
            double depth = 1.0;
            // x is the radius and y the axial coordinate, the result is the azimuthal vector potential A_phi
            // and J is the azimuthal current density, depth is not used
            bool axisymmetric = false;

            MagnetostaticSystem generateSystem(bool refine = true, double max_triangle_area = 1, int min_angle = 33, bool refine_magnets = false);
            std::vector<double> solve(MagnetostaticSystem& system);
            Vector computeForceIntegrals(std::vector<Vector> B, Vector p);
            double computeTorqueIntegral(std::vector<Vector> B, Vector p, Vector center);
            StressTensor computeStressIntegral(std::vector<Vector> B, Vector p, Vector center);
            Vector computeForceIntegrals(std::vector<Vector> B, SurroundingRegionBlockIntegralAssets assets, Vector p);
            double computeTorqueIntegral(std::vector<Vector> B, SurroundingRegionBlockIntegralAssets assets, Vector p, Vector center);
        protected:
            auto getSurroundingRegionBlockIntegralAssets(std::vector<Vector> B, Vector p);
    };
}

#endif