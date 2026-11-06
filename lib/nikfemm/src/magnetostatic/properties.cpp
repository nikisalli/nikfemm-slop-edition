#include <algorithm>
#include <cmath>
#include <cstdint>

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
        // secant permeability mu = B / H(B) on the smooth BH curve
        if (bh_curve.empty()) {
            return mu;
        }
        double nu, dnu_dB2;
        BHSpline(bh_curve).getNu(B * B, nu, dnu_dB2);
        return 1 / nu;
    }

    BHSpline::BHSpline(const BH_Curve& curve) {
        // the curve passes through the origin
        if (curve.empty() || curve.begin()->B > 0) {
            B.push_back(0);
            H.push_back(0);
        }
        for (auto& p : curve) {
            B.push_back(p.B);
            H.push_back(p.H);
        }
        if (B.size() < 2) {
            nexit("BH curve cannot have only one point");
        }
        computeSlopes();
    }

    void BHSpline::computeSlopes() {
        // same algorithm as femm's CMaterialProp::GetSlopes
        uint32_t n = B.size();
        for (uint32_t iter = 0; ; iter++) {
            // slopes of the spline with natural boundary conditions, tridiagonal system solved with the thomas algorithm
            std::vector<double> sub(n, 0), diag(n, 0), sup(n, 0), rhs(n, 0);
            double l = B[1] - B[0];
            diag[0] = 4 / l;
            sup[0] = 2 / l;
            rhs[0] = 6 * (H[1] - H[0]) / (l * l);
            l = B[n - 1] - B[n - 2];
            sub[n - 1] = 2 / l;
            diag[n - 1] = 4 / l;
            rhs[n - 1] = 6 * (H[n - 1] - H[n - 2]) / (l * l);
            for (uint32_t i = 1; i < n - 1; i++) {
                double l1 = B[i] - B[i - 1];
                double l2 = B[i + 1] - B[i];
                sub[i] = 2 / l1;
                diag[i] = 4 * (l1 + l2) / (l1 * l2);
                sup[i] = 2 / l2;
                rhs[i] = 6 * (H[i] - H[i - 1]) / (l1 * l1) + 6 * (H[i + 1] - H[i]) / (l2 * l2);
            }
            for (uint32_t i = 1; i < n; i++) {
                double w = sub[i] / diag[i - 1];
                diag[i] -= w * sup[i - 1];
                rhs[i] -= w * rhs[i - 1];
            }
            slope = std::vector<double>(n);
            slope[n - 1] = rhs[n - 1] / diag[n - 1];
            for (int64_t i = n - 2; i >= 0; i--) {
                slope[i] = (rhs[i] - sup[i] * slope[i + 1]) / diag[i];
            }

            // the curve is bad if dH/dB has a root inside a segment, i.e. H(B) is not monotonic
            bool curve_ok = true;
            for (uint32_t i = 1; i < n; i++) {
                double d0 = slope[i - 1];
                double d1 = slope[i];
                double u0 = H[i - 1];
                double u1 = H[i];
                double L = B[i] - B[i - 1];

                double c0 = d0;
                double c1 = -(2 * (2 * d0 * L + d1 * L + 3 * u0 - 3 * u1)) / (L * L);
                double c2 = (3 * (d0 * L + d1 * L + 2 * u0 - 2 * u1)) / (L * L * L);
                double X0 = -1;
                double X1 = -1;

                double disc = c1 * c1 - 4 * c0 * c2;
                if (c2 == 0) {
                    if (c1 != 0) X0 = -c0 / c1;
                } else if (disc > 0) {
                    disc = sqrt(disc);
                    X0 = -(c1 + disc) / (2 * c2);
                    X1 = (-c1 + disc) / (2 * c2);
                }
                if ((X0 >= 0 && X0 <= L) || (X1 >= 0 && X1 <= L)) {
                    curve_ok = false;
                }
            }
            if (curve_ok) {
                if (iter > 0) {
                    // femm does this silently, but it can change a coarse curve a lot
                    nlogwarn("the BH curve is not monotonic with a cubic spline, its points were smoothed %u times", iter);
                }
                break;
            }
            if (iter > 1000) {
                nexit("BH curve could not be made monotonic");
            }

            // smooth out the interior points with a 3 point moving average to get rid of rapid transitions
            std::vector<double> Bs = B;
            std::vector<double> Hs = H;
            for (uint32_t i = 1; i < n - 1; i++) {
                Bs[i] = (B[i - 1] + B[i] + B[i + 1]) / 3;
                Hs[i] = (H[i - 1] + H[i] + H[i + 1]) / 3;
            }
            B = Bs;
            H = Hs;
        }
    }

    void BHSpline::evaluate(double b, double& h, double& dhdb) const {
        b = fabs(b);
        uint32_t n = B.size();
        if (b >= B[n - 1]) {
            h = H[n - 1] + slope[n - 1] * (b - B[n - 1]);
            dhdb = slope[n - 1];
            return;
        }
        // segment [B[i], B[i + 1]] that contains b
        uint32_t i = std::upper_bound(B.begin(), B.end(), b) - B.begin() - 1;
        double l = B[i + 1] - B[i];
        double z = (b - B[i]) / l;
        double z2 = z * z;
        h = (1 - 3 * z2 + 2 * z2 * z) * H[i] +
            z * (1 - 2 * z + z2) * l * slope[i] +
            z2 * (3 - 2 * z) * H[i + 1] +
            z2 * (z - 1) * l * slope[i + 1];
        dhdb = 6 * z * (z - 1) * H[i] / l +
               (1 - 4 * z + 3 * z2) * slope[i] +
               6 * z * (1 - z) * H[i + 1] / l +
               z * (3 * z - 2) * slope[i + 1];
    }

    double BHSpline::getH(double b) const {
        double h, dhdb;
        evaluate(b, h, dhdb);
        return h;
    }

    double BHSpline::getdHdB(double b) const {
        double h, dhdb;
        evaluate(b, h, dhdb);
        return dhdb;
    }

    void BHSpline::getNu(double B2, double& nu, double& dnu_dB2) const {
        // same as femm's Get_v and Get_dvB2
        if (B2 <= 0) {
            nu = slope[0];
            dnu_dB2 = 0;
            return;
        }
        double b = sqrt(B2);
        double h, dhdb;
        evaluate(b, h, dhdb);
        nu = h / b;
        dnu_dB2 = 0.5 * (dhdb - nu) / B2;
    }

    bool MagnetostaticProp::operator==(const MagnetostaticProp& p) const {
        return mu == p.mu && J == p.J && M == p.M && bh_curve == p.bh_curve;
    }

    bool MagnetostaticProp::operator!=(const MagnetostaticProp& p) const {
        return !(*this == p);
    }
}