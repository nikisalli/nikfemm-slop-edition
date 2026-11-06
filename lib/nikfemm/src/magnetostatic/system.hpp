#ifndef NIK_MAGNETOSTATIC_SYSTEM_HPP
#define NIK_MAGNETOSTATIC_SYSTEM_HPP

#include <array>
#include <cstdint>
#include <vector>

namespace nikfemm {
    // finite element system of a magnetostatic simulation
    // the stiffness matrix depends on the reluctivity of the elements, so instead of the matrix this stores the
    // stiffness of every element with unit reluctivity and where each of its entries goes in the sparse matrix
    struct MagnetostaticSystem {
        // local node pairs of the 6 entries of the upper triangle of an element matrix
        static constexpr uint8_t elem_entry_nodes[6][2] = {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}};

        uint32_t n = 0;  // number of unknowns, one per node

        // sparsity pattern of the upper triangle of the stiffness matrix, compressed by column (row <= column)
        // the diagonal is the last entry of each column
        std::vector<int> col_ptr;
        std::vector<int> row_ind;

        // per element: the 6 entries of the stiffness matrix with unit reluctivity, the position of
        // each entry in row_ind and the element area
        std::vector<std::array<double, 6>> elem_stiffness;
        std::vector<std::array<int, 6>> elem_entries;
        std::vector<double> elem_area;

        std::vector<double> b;  // source vector (current densities and magnets)

        std::vector<uint32_t> dirichlet_nodes;
        std::vector<double> dirichlet_values;

        void addDirichletBoundaryCondition(uint32_t id, double value) {
            dirichlet_nodes.push_back(id);
            dirichlet_values.push_back(value);
        }
    };
}

#endif
