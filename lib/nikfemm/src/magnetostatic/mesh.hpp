#ifndef NIK_MAGNETOSTATICMESH_HPP
#define NIK_MAGNETOSTATICMESH_HPP

#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <chrono>
#include <math.h>
#include <set>

#include "../triangle/triangle.h"

#include "../utils/utils.hpp"
#include "../drawing/drawing.hpp"
#include "../mesh/mesh.hpp"
#include "../algebra/coo.hpp"
#include "../algebra/system.hpp"
#include "system.hpp"
#include "properties.hpp"

namespace nikfemm {
    struct MagnetostaticMesh : Mesh<MagnetostaticProp> {
        bool axisymmetric = false;

        MagnetostaticMesh();

        MagnetostaticSystem getFemSystem();
        void addDirichletInfiniteBoundaryConditions(MagnetostaticSystem& system);
        void refineMeshAroundMagnets();
        // B = curl(A), in axisymmetric problems (B_r, B_z) from the azimuthal vector potential
        std::vector<Vector> computeCurl(std::vector<double>& in) const;
    };
}

#endif