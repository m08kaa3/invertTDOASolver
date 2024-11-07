#include "InternalHeaderCheck.h"
#include "LevenbergMarquardtRootFinder.h"

namespace rootfinder
{

template<typename TargetFunction>
void LevenbergMarquardtRootFinder<TargetFunction>::setMaximalDampingFactor(
    double maximalDampingFactor)
{
    assert(maximalDampingFactor > 0);
    maximalDampingFactor_ = maximalDampingFactor;
}


template<typename TargetFunction>
bool LevenbergMarquardtRootFinder<TargetFunction>::stepImpl(
    double minimalImprovement)
{
    Eigen::JacobiSVD<Eigen::Matrix<double, size, size>> svd(currentDf_, 
        Eigen::ComputeFullU | Eigen::ComputeFullV);
    const auto& Lambdas = svd.singularValues().array();
    const auto ones = Eigen::Matrix<double, size, 1>::Ones().array();
    const auto v = svd.matrixU().transpose() * currentFMinusC_;
    while (dampingFactor_ <= maximalDampingFactor_)
    {
        const Eigen::Matrix<double, size, 1> middleMat = (Lambdas.array() /
            (Lambdas * Lambdas + dampingFactor_ * ones)).matrix();
        const Eigen::Matrix<double, size, 1> correction = 
            -svd.matrixV() * middleMat.asDiagonal() * v;
        const Eigen::Matrix<double, size, 1> newFMinusC = 
            f_.F(currentZ_ + correction) - c_;
        const double newResidual = newFMinusC.norm();
        if (newResidual < currentResidual_ - minimalImprovement)
        {
            dampingFactor_ /= 10.;
            currentZ_ += correction;
            currentFMinusC_ = newFMinusC;
            currentResidual_ = newResidual;
            currentDf_ = f_.DF(currentZ_);
            return true;
        }
        else
        {
            dampingFactor_ *= 10.;
        }
    }
    return false;
}

} // namespace rootfinder