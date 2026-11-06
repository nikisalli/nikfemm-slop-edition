#include <cmath>
#include <iterator>

#include "properties.hpp"
#include "../utils/utils.hpp"

namespace nikfemm {
    bool BH_Point::operator==(const BH_Point& p) const {
        return this->B == p.B && this->H == p.H;
    }

    bool MagnetostaticProp::isLinear() const {
        return bh_curve.size() == 0;
    }

    double MagnetostaticProp::getMu(double B) const {
        // secant permeability mu = B / H(B), which is what the fixed point iteration on
        // div(1/mu grad A) = -J needs (not the differential permeability dB/dH)
        // H(B) is linearly interpolated between the curve points, the curve is assumed
        // to pass through the origin and is extrapolated past the last point with the slope of the last segment
        if (bh_curve.empty()) {
            return mu;
        } else if (bh_curve.size() == 1) {
            nexit("BH curve cannot have only one point");
        }
        B = fabs(B);

        // find the segment [B1, B2] that contains B
        double B1, H1, B2, H2;
        BH_Curve::const_iterator it = bh_curve.upper_bound({B, 0});
        if (it == bh_curve.begin()) {
            // B is smaller than the smallest B in the curve, use the segment from the origin
            B1 = 0;
            H1 = 0;
            B2 = it->B;
            H2 = it->H;
        } else if (it == bh_curve.end()) {
            // B is larger than the largest B in the curve, extrapolate the last segment
            B2 = std::prev(it, 1)->B;
            H2 = std::prev(it, 1)->H;
            B1 = std::prev(it, 2)->B;
            H1 = std::prev(it, 2)->H;
        } else {
            B1 = std::prev(it, 1)->B;
            H1 = std::prev(it, 1)->H;
            B2 = it->B;
            H2 = it->H;
        }

        double H = H1 + (H2 - H1) * (B - B1) / (B2 - B1);
        if (H <= 0) {
            // B = 0 on a segment that starts at the origin, the secant permeability tends to the slope of the segment
            return (B2 - B1) / (H2 - H1);
        }
        return B / H;
    }

    bool MagnetostaticProp::operator==(const MagnetostaticProp& p) const {
        return mu == p.mu && J == p.J && M == p.M && bh_curve == p.bh_curve;
    }

    bool MagnetostaticProp::operator!=(const MagnetostaticProp& p) const {
        return !(*this == p);
    }
}