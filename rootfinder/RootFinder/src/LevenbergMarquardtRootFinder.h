#ifndef LEVENBERG_MARQUARDT_ROOT_FINDER_H
#define LEVENBERG_MARQUARDT_ROOT_FINDER_H

#include "InternalHeaderCheck.h"

namespace rootfinder
{

/**
 * @brief Implementation of the Levenberg-Marquardt root-finding algorithm.
 * 
 * This algorithm combines the Newton method and gradient descent, using a  
 * parameter \f$ \lambda \f$, called the damping factor. For 
 * \f$ \lambda \approx 0 \f$, this algorithm closely resembles the Newton 
 * method; with a large damping factor, it approaches gradient descent with a 
 * learning rate of \f$ \lambda^{-1} \f$.
 */
template<typename TargetFunction>
class LevenbergMarquardtRootFinder : 
    public internal::BaseRootFinder<
        LevenbergMarquardtRootFinder<TargetFunction>, TargetFunction>
{
    using Base = internal::BaseRootFinder<
        LevenbergMarquardtRootFinder<TargetFunction>, TargetFunction>;
    friend Base;
	using Base::f_;
	using Base::c_;
	using Base::currentZ_;
    using Base::currentFMinusC_;
    using Base::currentResidual_;
    using Base::currentDf_;
    using Base::size;

public:
    using Base::Base;

public:
    /**
     * @brief Sets the maximum allowed damping factor \f$ \lambda \f$.
     * @param maximalDampingFactor The maximum allowed damping factor.
     */
    void setMaximalDampingFactor(double maximalDampingFactor);

protected:
    /**
     * @brief Step implementation used in `BaseRootFinder`.
     */
    bool stepImpl(double minimalImprovement);

    /**
     * @brief Reset additional parameters implementation used in 
     * `BaseRootFinder`.
     */
	void resetAdditionalParametersImpl() {dampingFactor_ = 1e-2;}

private:
    // Damping factor.
    double dampingFactor_ = 1e-2;
    // Maximum allowed damping factor.
    double maximalDampingFactor_ = 1e+5;
};

} // namespace rootfinder

#include "LevenbergMarquardtRootFinder.ipp"

#endif // LEVENBERG_MARQUARDT_ROOT_FINDER_H