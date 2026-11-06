#include <nikfemm.hpp>

#define BOOST_TEST_MODULE VsidCommonTest
#include <boost/test/unit_test.hpp>

using namespace nikfemm;

BOOST_AUTO_TEST_SUITE( csr_tests )

BOOST_AUTO_TEST_CASE( csr_index )
{
    MatCOOSymmetric<double> mat(3);

    mat(0, 0) = 1;
    mat(0, 1) = 2;
    mat(0, 2) = 3;
    mat(1, 1) = 4;
    mat(1, 2) = 5;
    mat(2, 2) = 6;

    MatCSRSymmetric csr(mat);

    BOOST_CHECK_EQUAL(csr(0, 0), 1);
    BOOST_CHECK_EQUAL(csr(0, 1), 2);
    BOOST_CHECK_EQUAL(csr(0, 2), 3);
    BOOST_CHECK_EQUAL(csr(1, 1), 4);
    BOOST_CHECK_EQUAL(csr(1, 2), 5);
    BOOST_CHECK_EQUAL(csr(2, 2), 6);
}

BOOST_AUTO_TEST_CASE( csr_cv_mult )
{
    MatCOOSymmetric<double> mat(3);

    mat(0, 0) = 1;
    mat(0, 1) = 2;
    mat(0, 2) = 3;
    mat(1, 1) = 4;
    mat(1, 2) = 5;
    mat(2, 2) = 6;

    MatCSRSymmetric csr(mat);

    std::vector<double> x(3);
    x[0] = 1;
    x[1] = 2;
    x[2] = 3;

    std::vector<double> y(3);

    mult(y, csr, x);

    // wolfram alpha: {{1, 2, 3}, {2, 4, 5}, {3, 5, 6}} * {1, 2, 3}

    BOOST_CHECK_EQUAL(y[0], 14);
    BOOST_CHECK_EQUAL(y[1], 25);
    BOOST_CHECK_EQUAL(y[2], 31);
}

BOOST_AUTO_TEST_CASE( csr_conjugate_gradient_solve )
{
    MatCOOSymmetric<double> mat(3);

    mat(0, 0) = 1;
    mat(0, 1) = 2;
    mat(0, 2) = 3;
    mat(1, 1) = 4;
    mat(1, 2) = 5;
    mat(2, 2) = 6;

    MatCSRSymmetric csr(mat);

    std::vector<double> x(3);
    x[0] = 0;
    x[1] = 0;
    x[2] = 0;

    std::vector<double> b(3);
    b[0] = 14;
    b[1] = 32;
    b[2] = 54;

    preconditionedSSORConjugateGradientSolver(csr, b, x, 1.5, 1e-6, 1000);

    // wolfram alpha: solve {{1, 2, 3}, {2, 4, 5}, {3, 5, 6}} * {x, y, z} = {14, 32, 54}
    // solution: {x, y, z} = {26, 0, -4}

    // check result inside tolerance
    BOOST_CHECK_SMALL(x[0] - 26, 1e-6);
    BOOST_CHECK_SMALL(x[1] - 0, 1e-6);
    BOOST_CHECK_SMALL(x[2] - -4, 1e-6);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE( cv_tests )

BOOST_AUTO_TEST_CASE( cv_index )
{
    std::vector<double> cv(3);
    cv[0] = 1;
    cv[1] = 2;
    cv[2] = 3;

    BOOST_CHECK_EQUAL(cv[0], 1);
    BOOST_CHECK_EQUAL(cv[1], 2);
    BOOST_CHECK_EQUAL(cv[2], 3);
}

BOOST_AUTO_TEST_CASE( cv_mult )
{
    std::vector<double> x(3);
    x[0] = 1;
    x[1] = 2;
    x[2] = 3;

    std::vector<double> y(3);
    y[0] = 4;
    y[1] = 5;
    y[2] = 6;

    std::vector<double> z(3);
    
    mult(z, x, y);

    BOOST_CHECK_EQUAL(z[0], 4);
    BOOST_CHECK_EQUAL(z[1], 10);
    BOOST_CHECK_EQUAL(z[2], 18);
}

BOOST_AUTO_TEST_CASE( cv_mult_scalar )
{
    std::vector<double> x(3);
    x[0] = 1;
    x[1] = 2;
    x[2] = 3;

    std::vector<double> y(3);
    
    mult(y, 2, x);

    BOOST_CHECK_EQUAL(y[0], 2);
    BOOST_CHECK_EQUAL(y[1], 4);
    BOOST_CHECK_EQUAL(y[2], 6);
}

BOOST_AUTO_TEST_CASE( cv_add )
{
    std::vector<double> x(3);
    x[0] = 1;
    x[1] = 2;
    x[2] = 3;

    std::vector<double> y(3);
    y[0] = 4;
    y[1] = 5;
    y[2] = 6;

    std::vector<double> z(3);
    
    add(z, x, y);

    BOOST_CHECK_EQUAL(z[0], 5);
    BOOST_CHECK_EQUAL(z[1], 7);
    BOOST_CHECK_EQUAL(z[2], 9);
}

BOOST_AUTO_TEST_CASE( cv_sub )
{
    std::vector<double> x(3);
    x[0] = 1;
    x[1] = 2;
    x[2] = 3;

    std::vector<double> y(3);
    y[0] = 4;
    y[1] = 5;
    y[2] = 6;

    std::vector<double> z(3);
    
    sub(z, x, y);

    BOOST_CHECK_EQUAL(z[0], -3);
    BOOST_CHECK_EQUAL(z[1], -3);
    BOOST_CHECK_EQUAL(z[2], -3);
}

BOOST_AUTO_TEST_CASE( cv_div )
{
    std::vector<double> x(3);
    x[0] = 1;
    x[1] = 2;
    x[2] = 3;

    std::vector<double> y(3);
    y[0] = 4;
    y[1] = 5;
    y[2] = 6;

    std::vector<double> z(3);
    
    div(z, x, y);

    BOOST_CHECK_EQUAL(z[0], 0.25);
    BOOST_CHECK_EQUAL(z[1], 0.4);
    BOOST_CHECK_EQUAL(z[2], 0.5);
}

BOOST_AUTO_TEST_CASE( cv_div_scalar )
{
    std::vector<double> x(3);
    x[0] = 1;
    x[1] = 2;
    x[2] = 3;

    std::vector<double> y(3);
    
    div(y, x, 2);

    BOOST_CHECK_EQUAL(y[0], 0.5);
    BOOST_CHECK_EQUAL(y[1], 1);
    BOOST_CHECK_EQUAL(y[2], 1.5);
}

BOOST_AUTO_TEST_CASE( cv_sub_scalar )
{
    std::vector<double> x(3);
    x[0] = 1;
    x[1] = 2;
    x[2] = 3;

    std::vector<double> y(3);
    
    sub(y, x, 2);

    BOOST_CHECK_EQUAL(y[0], -1);
    BOOST_CHECK_EQUAL(y[1], 0);
    BOOST_CHECK_EQUAL(y[2], 1);
}

BOOST_AUTO_TEST_CASE( cv_add_scalar )
{
    std::vector<double> x(3);
    x[0] = 1;
    x[1] = 2;
    x[2] = 3;

    std::vector<double> y(3);
    
    add(y, x, 2);

    BOOST_CHECK_EQUAL(y[0], 3);
    BOOST_CHECK_EQUAL(y[1], 4);
    BOOST_CHECK_EQUAL(y[2], 5);
}

BOOST_AUTO_TEST_CASE( cv_dot )
{
    std::vector<double> x(3);
    x[0] = 1;
    x[1] = 2;
    x[2] = 3;

    std::vector<double> y(3);
    y[0] = 4;
    y[1] = 5;
    y[2] = 6;

    double z = dot(x, y);

    BOOST_CHECK_EQUAL(z, 32);
}

BOOST_AUTO_TEST_CASE( cv_add_scaled )
{
    std::vector<double> x(3);
    x[0] = 1;
    x[1] = 2;
    x[2] = 3;

    std::vector<double> y(3);
    y[0] = 4;
    y[1] = 5;
    y[2] = 6;

    std::vector<double> z(3);
    
    addScaled(z, x, 2, y);

    BOOST_CHECK_EQUAL(z[0], 9);
    BOOST_CHECK_EQUAL(z[1], 12);
    BOOST_CHECK_EQUAL(z[2], 15);
}

BOOST_AUTO_TEST_CASE( cv_square_sum )
{
    std::vector<double> x(3);
    x[0] = 1;
    x[1] = 2;
    x[2] = 3;

    double z = squareSum(x);

    BOOST_CHECK_EQUAL(z, 14);
}

BOOST_AUTO_TEST_CASE( cv_copy )
{
    std::vector<double> x(3);
    x[0] = 1;
    x[1] = 2;
    x[2] = 3;

    std::vector<double> y(3);
    
    copy(y, x);

    BOOST_CHECK_EQUAL(y[0], 1);
    BOOST_CHECK_EQUAL(y[1], 2);
    BOOST_CHECK_EQUAL(y[2], 3);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE( current_density_tests )

BOOST_AUTO_TEST_CASE( current_density_dirichlet_boundary_conditions )
{
    MatCOOSymmetric<double> mat(3);

    mat(0, 0) = 1;
    mat(0, 1) = 2;
    mat(0, 2) = 3;
    mat(1, 1) = 4;
    mat(1, 2) = 5;
    mat(2, 2) = 6;

    // 1, 2, 3
    // 2, 4, 5
    // 3, 5, 6

    std::vector<double> b(3);
    b[0] = 14;
    b[1] = 32;
    b[2] = 54;

    System<double> system(mat, b);

    printf("A without Dirichlet boundary conditions:\n");
    MatCSRSymmetric csr_1(system.A);
    csr_1.print();
    printf("b without Dirichlet boundary conditions:\n");
    for (auto const& elem : system.b) {
        printf("%.17g\n", elem);
    }

    system.addDirichletBoundaryCondition(0, 1.0);
    system.addDirichletBoundaryCondition(1, 2.0);

    MatCSRSymmetric csr(system.A);

    printf("A with Dirichlet boundary conditions:\n");
    csr.print();
    printf("b with Dirichlet boundary conditions:\n");
    for (auto const& elem : system.b) {
        printf("%.17g\n", elem);
    }

    std::vector<double> x(3);
    x[0] = 0;
    x[1] = 0;
    x[2] = 0;

    preconditionedSSORConjugateGradientSolver(csr, system.b, x, 1.5, 1e-10, 1000);

    // wolfram alpha: solve {{1, 0, 0}, {0, 1, 0}, {0, 0, 6}} * {x, y, z} = {1, 2, 41}

    // 1, 0, 0
    // 0, 4, 5
    // 0, 5, 6

    // solution: {x, y, z} = {1, 75, -54}

    // check result inside tolerance
    BOOST_CHECK_SMALL(x[0] - 1, 1e-6);
    BOOST_CHECK_SMALL(x[1] - 2, 1e-6);
    BOOST_CHECK_SMALL(x[2] - 41.0/6.0, 1e-6);
}

BOOST_AUTO_TEST_SUITE_END()
// analytic magnetostatic benchmarks, SI units
namespace {
    const double MU0 = 4 * PI * 1e-7;

    // area weighted average of the azimuthal component of B over the elements whose barycenter is at distance [r - dr, r + dr] from c
    double ringAverageBtheta(MagnetostaticSimulation& sim, std::vector<Vector>& B, Vector c, double r, double dr) {
        auto& d = sim.mesh.data;
        double w = 0, bt = 0;
        for (int i = 0; i < d.numberoftriangles; i++) {
            Vector p = d.getElemBarycenter(i) + sim.mesh.center - c;
            if (fabs(p.norm() - r) > dr) continue;
            double th = atan2(p.y, p.x);
            bt += d.getArea(i) * (-B[i].x * sin(th) + B[i].y * cos(th));
            w += d.getArea(i);
        }
        return bt / w;
    }

    // coaxial cable: inner conductor +I, gap material, outer return conductor -I
    // by symmetry H = I / (2 pi r) in the gap whatever the material is
    std::vector<Vector> solveCoax(MagnetostaticSimulation& sim, MagnetostaticProp gap, double I) {
        double a = 0.01, b = 0.03, c = 0.035;
        sim.mesh.drawing.drawCircle(Vector(0, 0), a, 120);
        sim.mesh.drawing.drawCircle(Vector(0, 0), b, 120);
        sim.mesh.drawing.drawCircle(Vector(0, 0), c, 120);
        sim.mesh.drawing.drawRegion(Vector(0, 0), {I / (PI * a * a), {0, 0}, MU0});
        sim.mesh.drawing.drawRegion(Vector(0.02, 0.0005), gap);
        sim.mesh.drawing.drawRegion(Vector(0.0325, 0.0005), {-I / (PI * (c * c - b * b)), {0, 0}, MU0});
        auto system = sim.generateSystem(false, 1e-6);
        auto A = sim.solve(system);
        return sim.mesh.computeCurl(A);
    }
}

BOOST_AUTO_TEST_SUITE( magnetostatic_tests )

BOOST_AUTO_TEST_CASE( magnetostatic_magnetized_cylinder )
{
    // uniformly magnetized cylinder with mu_r = 1: B inside = mu0 M / 2, the test is repeated in metres and millimetres
    for (double scale : {1.0, 1000.0}) {
        MagnetostaticSimulation sim;
        double a = 0.01 * scale, M = 1e6;
        sim.mesh.drawing.drawCircle(Vector(0, 0), a, 120);
        sim.mesh.drawing.drawRegion(Vector(0, 0), {0, {M, 0}, MU0});
        auto system = sim.generateSystem(false, 1e-6 * scale * scale);
        auto A = sim.solve(system);
        auto B = sim.mesh.computeCurl(A);

        auto& d = sim.mesh.data;
        Vector acc = {0, 0};
        double w = 0;
        for (int i = 0; i < d.numberoftriangles; i++) {
            if ((d.getElemBarycenter(i) + sim.mesh.center).norm() > 0.8 * a) continue;
            acc += B[i] * d.getArea(i);
            w += d.getArea(i);
        }
        BOOST_CHECK_CLOSE(acc.x / w, MU0 * M / 2, 0.5);
        BOOST_CHECK_SMALL(acc.y / w, 1e-3);
    }
}

BOOST_AUTO_TEST_CASE( magnetostatic_coax_linear )
{
    double I = 100;
    for (double mu_r : {1.0, 1000.0}) {
        MagnetostaticSimulation sim;
        auto B = solveCoax(sim, {0, {0, 0}, mu_r * MU0}, I);
        for (double r : {0.015, 0.02, 0.025}) {
            BOOST_CHECK_CLOSE(ringAverageBtheta(sim, B, {0, 0}, r, 0.001), mu_r * MU0 * I / (2 * PI * r), 0.5);
        }
    }
}

BOOST_AUTO_TEST_CASE( magnetostatic_coax_nonlinear )
{
    // B in the gap must follow the BH curve at H = I / (2 pi r), the radii avoid the corners of the piecewise linear curve
    double I = 400;
    MagnetostaticSimulation sim;
    auto B = solveCoax(sim, {0, {0, 0}, 0, magnetostatic_materials::iron}, I);
    for (double r : {0.016, 0.02, 0.025}) {
        double H = I / (2 * PI * r);
        double Bexpected = H < 2000 ? H / 2000 : (H < 5000 ? 1 + 0.42 * (H - 2000) / 3000 : 1.42 + 0.28 * (H - 5000) / 5000);
        BOOST_CHECK_CLOSE(ringAverageBtheta(sim, B, {0, 0}, r, 0.001), Bexpected, 0.5);
    }
}

BOOST_AUTO_TEST_CASE( magnetostatic_off_center_wire )
{
    // a single wire carrying a net current, away from the center of the boundary: the return current is at infinity
    double a = 0.01, I = 100;
    Vector c0 = {0.3, -0.2};
    MagnetostaticSimulation sim;
    sim.mesh.drawing.drawCircle(c0, a, 120);
    sim.mesh.drawing.drawCircle(c0 + Vector(0.05, 0), a, 120);
    sim.mesh.drawing.drawRegion(c0, {I / (PI * a * a), {0, 0}, MU0});
    sim.mesh.drawing.drawRegion(c0 + Vector(0.05, 0), {0, {0, 0}, MU0});
    auto system = sim.generateSystem(false, 1e-6);
    auto A = sim.solve(system);
    auto B = sim.mesh.computeCurl(A);
    for (double r : {0.015, 0.02}) {
        BOOST_CHECK_CLOSE(ringAverageBtheta(sim, B, c0, r, 0.001), MU0 * I / (2 * PI * r), 0.5);
    }
}

BOOST_AUTO_TEST_SUITE_END()
