#ifndef BASE_ROOT_FINDER_H
#define BASE_ROOT_FINDER_H

#include "InternalHeaderCheck.h"

namespace rootfinder
{
namespace internal
{

/**
 * @brief A base class for one-step root-finding algorithms.
 * 
 * This class provides an interface for solving equations of the form 
 * \f$ F(z) = c \f$, where \f$ F : \mathbb{R}^N \rightarrow \mathbb{R}^N \f$, 
 * using iterative approximations.
 * 
 * To create a custom algorithm, inherit from this class and implement the 
 * method `bool stepImpl(double minimalImprovement)`. The parameter 
 * `minimalImprovement` represents the minimum allowed improvement per step.
 * 
 * Example:
 * @code 
 * template<typename TargetFunction>
 * class MyRootFinder : 
 *     public BaseRootFinder<MyRootFinder<TargetFunction>, TargetFunction>
 * {
 *     using Base = BaseRootFinder<MyRootFinder<TargetFunction>, TargetFunction>;
 *     friend Base;
 * protected:
 *     bool stepImpl(double minimalImprovement) {...}
 * };
 * @endcode
 * 
 * @note `BaseRootFinder` and all its derived classes store a non-owning
 * reference to the `TargetFunction` instance. This instance should not be
 * modified during the root-finding process.
 */
template<typename DerivedRootFinder, typename TargetFunction>
class BaseRootFinder
{
public:
	// Size of the input/output vectors of the target function.
	static constexpr size_t size = BaseTargetFunction<TargetFunction>::size();
public:
    /**
     * @brief Constructor.
     * @param f A `TargetFunction` instance.
     * @param c The right-hand side vector.
     */
    template<typename EigenDerived>
	BaseRootFinder(const BaseTargetFunction<TargetFunction>& f, 
		const Eigen::MatrixBase<EigenDerived>& c);

    /**
     * @brief Resets all computations and generates a new initial value.
     */
    void reset();

	/**
	 * @brief Gets the current step residual.
	 * @return The current residual value.
	 */
	double getResidual() const;

    /**
     * @brief Gets the current approximation of the root.
     * @return The current root approximation.
     */
    const Eigen::Matrix<double, size, 1>& getCurrentZ() const { return currentZ_; }

    /**
     * @brief Performs a single step of the algorithm.
     * 
     * @param minimalImprovement The minimum allowed improvement. If a new root
     * approximation yields an improvement less than `minimalImprovement`, 
     * the algorithm stops.
     * @return `true` if a new root approximation was found; otherwise, `false`.
     */
    bool step(double minimalImprovement = 1e-15);

protected: // main parameters
	// Target function instance.
	const BaseTargetFunction<TargetFunction>& f_;
	// Right-hand side vector.
	Eigen::Matrix<double, size, 1> c_;
	// Current approximation of the root.
	Eigen::Matrix<double, size, 1> currentZ_;
protected: // Auxiliary parameters
    // F(currentZ_) - c_
	Eigen::Matrix<double, size, 1> currentFMinusC_;
    // The norm of currentFMinusC_.
	double currentResidual_;
    // Jacobian of F at the currentZ_.
	Eigen::Matrix<double, size, size> currentDf_;
};


} // namespace internal
} // namespace rootfinder

#include "BaseRootFinder.ipp"

#endif //BASE_ROOT_FINDER_H