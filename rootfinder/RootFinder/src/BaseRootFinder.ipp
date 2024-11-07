#include "InternalHeaderCheck.h"
#include "BaseRootFinder.h"

namespace rootfinder
{
namespace internal
{


template<typename DerivedRootFinder, typename TargetFunction>
template<typename EigenDerived>
BaseRootFinder<DerivedRootFinder, TargetFunction>::BaseRootFinder(
    const BaseTargetFunction<TargetFunction>& f, 
    const Eigen::MatrixBase<EigenDerived>& c) :
    f_{f}, c_{c}
{
    f_.checkArgumentCorrectness(c_);
    reset();
}


template<typename DerivedRootFinder, typename TargetFunction>
void BaseRootFinder<DerivedRootFinder, TargetFunction>::reset()
{
    currentZ_ = f_.initialValue();
    currentFMinusC_ = f_.F(currentZ_) - c_;
    currentResidual_ = currentFMinusC_.norm();
    currentDf_ = f_.DF(currentZ_);
	static_cast<DerivedRootFinder*>(this)->resetAdditionalParametersImpl();
}


template<typename DerivedRootFinder, typename TargetFunction>
double BaseRootFinder<DerivedRootFinder, TargetFunction>::getResidual() const 
{ return currentResidual_; }


template<typename DerivedRootFinder, typename TargetFunction>
bool BaseRootFinder<DerivedRootFinder, TargetFunction>::step(
    double minimalImprovement/* = 1e-15*/)
{
    return static_cast<DerivedRootFinder*>(this)->
        stepImpl(minimalImprovement);
}

} // namespace internal
} // namespace rootfinder