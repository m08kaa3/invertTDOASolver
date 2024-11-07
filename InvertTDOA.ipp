#include "InvertTDOA.h"


InvertTDOA::InvertTDOA(const Eigen::Vector2d& x1, const Eigen::Vector2d& x2, 
    const Eigen::Vector2d& x3) : x1_(x1), x2_(x2), x3_(x3) 
{}


template <typename EigenDerived>
Eigen::Matrix<double, InvertTDOA::size, 1> 
InvertTDOA::Fimpl(const Eigen::MatrixBase<EigenDerived>& z) const
{
    const auto& y1 = z.block(0,0,2,1);
    const auto& y2 = z.block(2,0,2,1);
    const auto& y3 = z.block(4,0,2,1);

    Eigen::Matrix<double, size, 1> result;
    result << (y1 - x1_).norm() - (y2 - x1_).norm(),
                (y2 - x1_).norm() - (y3 - x1_).norm(),
                (y1 - x2_).norm() - (y2 - x2_).norm(),
                (y2 - x2_).norm() - (y3 - x2_).norm(),
                (y1 - x3_).norm() - (y2 - x3_).norm(),
                (y2 - x3_).norm() - (y3 - x3_).norm();
    return result;
}


template <typename EigenDerived>
Eigen::Matrix<double, InvertTDOA::size, InvertTDOA::size> 
InvertTDOA::DFimpl(const Eigen::MatrixBase<EigenDerived>& z) const
{
    const auto& y1 = z.block(0,0,2,1);
    const auto& y2 = z.block(2,0,2,1);
    const auto& y3 = z.block(4,0,2,1);

    Eigen::Matrix<double, size, size> result;
    result(0,0) =  (y1(0,0) - x1_(0,0)) / (y1 - x1_).norm();
    result(0,1) =  (y1(1,0) - x1_(1,0)) / (y1 - x1_).norm();
    result(0,2) = -(y2(0,0) - x1_(0,0)) / (y2 - x1_).norm();
    result(0,3) = -(y2(1,0) - x1_(1,0)) / (y2 - x1_).norm();
    result(0,4) =                                         0;
    result(0,5) =                                         0;
    result(1,0) =                                         0;
    result(1,1) =                                         0;
    result(1,2) =  (y2(0,0) - x1_(0,0)) / (y2 - x1_).norm();
    result(1,3) =  (y2(1,0) - x1_(1,0)) / (y2 - x1_).norm();
    result(1,4) = -(y3(0,0) - x1_(0,0)) / (y3 - x1_).norm();
    result(1,5) = -(y3(1,0) - x1_(1,0)) / (y3 - x1_).norm();
    result(2,0) =  (y1(0,0) - x2_(0,0)) / (y1 - x2_).norm();
    result(2,1) =  (y1(1,0) - x2_(1,0)) / (y1 - x2_).norm();
    result(2,2) = -(y2(0,0) - x2_(0,0)) / (y2 - x2_).norm();
    result(2,3) = -(y2(1,0) - x2_(1,0)) / (y2 - x2_).norm();
    result(2,4) =                                         0;
    result(2,5) =                                         0;
    result(3,0) =                                         0;
    result(3,1) =                                         0;
    result(3,2) =  (y2(0,0) - x2_(0,0)) / (y2 - x2_).norm();
    result(3,3) =  (y2(1,0) - x2_(1,0)) / (y2 - x2_).norm();
    result(3,4) = -(y3(0,0) - x2_(0,0)) / (y3 - x2_).norm();
    result(3,5) = -(y3(1,0) - x2_(1,0)) / (y3 - x2_).norm();
    result(4,0) =  (y1(0,0) - x3_(0,0)) / (y1 - x3_).norm();
    result(4,1) =  (y1(1,0) - x3_(1,0)) / (y1 - x3_).norm();
    result(4,2) = -(y2(0,0) - x3_(0,0)) / (y2 - x3_).norm();
    result(4,3) = -(y2(1,0) - x3_(1,0)) / (y2 - x3_).norm();
    result(4,4) =                                         0;
    result(4,5) =                                         0;
    result(5,0) =                                         0;
    result(5,1) =                                         0;
    result(5,2) =  (y2(0,0) - x3_(0,0)) / (y2 - x3_).norm();
    result(5,3) =  (y2(1,0) - x3_(1,0)) / (y2 - x3_).norm();
    result(5,4) = -(y3(0,0) - x3_(0,0)) / (y3 - x3_).norm();
    result(5,5) = -(y3(1,0) - x3_(1,0)) / (y3 - x3_).norm();

    return result;
}


Eigen::Matrix<double, InvertTDOA::size, 1> InvertTDOA::initialValueimpl() const
{
    const Eigen::Vector2d barycenter = (x1_ + x2_ + x3_)/3.;
    const double radius = std::max({
        (x1_ - barycenter).norm(), 
        (x2_ - barycenter).norm(), 
        (x3_ - barycenter).norm()});

    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<double> d{0., 1.0};
    
    Eigen::Matrix<double, size, 1> result;
    result << d(gen)*radius + barycenter(0), 
                d(gen)*radius + barycenter(1), 
                d(gen)*radius + barycenter(0), 
                d(gen)*radius + barycenter(1), 
                d(gen)*radius + barycenter(0), 
                d(gen)*radius + barycenter(1);
    return result;
}