#ifndef NIK_MAGNETOSTATIC_PROPERTIES_HPP
#define NIK_MAGNETOSTATIC_PROPERTIES_HPP

#include <set>
#include <vector>

#include "../geometry/vector.hpp"

namespace nikfemm {
    struct BH_Point {
        double B;
        double H;

        bool operator==(const BH_Point& p) const;
        bool operator!=(const BH_Point& p) const;
    };
}

template <>
struct std::less<nikfemm::BH_Point> {
    inline bool operator()(const nikfemm::BH_Point& p1, const nikfemm::BH_Point& p2) const {
        return p1.B < p2.B;
    }
};

template <>
struct std::equal_to<nikfemm::BH_Point> {
    inline bool operator()(const nikfemm::BH_Point& p1, const nikfemm::BH_Point& p2) const {
        return p1.B == p2.B;
    }
};

namespace nikfemm {
    typedef std::set<BH_Point> BH_Curve;

    // smooth interpolation of a BH curve, same as femm: H(B) is a cubic hermite spline through the curve points
    // and the origin, extrapolated linearly past the last point. if the spline is not monotonic the points
    // are smoothed until it is
    struct BHSpline {
        std::vector<double> B;
        std::vector<double> H;
        std::vector<double> slope;  // dH/dB at the points

        BHSpline() {}
        BHSpline(const BH_Curve& curve);
        double getH(double b) const;
        double getdHdB(double b) const;
        // reluctivity nu = H / B and its derivative with respect to B^2
        void getNu(double B2, double& nu, double& dnu_dB2) const;

        private:
        void computeSlopes();
        void evaluate(double b, double& h, double& dhdb) const;
    };

    typedef double MaxwellStressTensor[2][2];

    struct MagnetostaticProp {
        double J; // current density
        Vector M; // magnetization
        double mu; // permeability
        BH_Curve bh_curve; // BH curve

        double getMu(double B) const;
        bool isLinear() const;

        bool operator==(const MagnetostaticProp& p) const;
        bool operator!=(const MagnetostaticProp& p) const;
    };

    // default material property
    namespace magnetostatic_materials {
        // specify mu or bh_curve, if both the latter is used
        const double vacuum = 4 * PI * 1e-7;
        const double air = 4 * PI * 1e-7;
        const double copper = 1.256629e-6;
        const BH_Curve iron = {{1, 2000}, {1.42, 5000}, {1.7, 10000}, {2, 40000}};
        const double iron_linear = 6.3e-3;
        const double neodymium = 1.32e-6;
    }
}

template <>
struct std::hash<nikfemm::MagnetostaticProp> {
    inline std::size_t operator()(const nikfemm::MagnetostaticProp& p) const {
        return std::hash<double>()(p.mu) ^ std::hash<double>()(p.J) ^ std::hash<nikfemm::Vector>()(p.M);
    }
};

template <>
struct std::equal_to<nikfemm::MagnetostaticProp> {
    inline bool operator()(const nikfemm::MagnetostaticProp& p1, const nikfemm::MagnetostaticProp& p2) const {
        return p1.mu == p2.mu && p1.J == p2.J && p1.M == p2.M && p1.bh_curve == p2.bh_curve;
    }
};

#endif