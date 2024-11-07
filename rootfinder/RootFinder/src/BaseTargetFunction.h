#ifndef BASE_TARGET_FUNCTION_H
#define BASE_TARGET_FUNCTION_H

#include "InternalHeaderCheck.h"

namespace rootfinder
{
/**
 * @brief A base class describing target function for root-finding algorithms.
 * 
 * To create your own target function, called `DerivedTargetFunction`, follow 
 * these steps:
 * 1. Inherit `DerivedTargetFunction` from `BaseTargetFunction<DerivedTargetFunction>`.
 * 2. Implement the method `Fimpl` that defines a target function 
 *    \f$ F: \mathbb{R}^N \rightarrow \mathbb{R}^N \f$. 
 * 3. Add the required field `size` -- the size of the input and output vectors, 
 *    denoted \f$ N \f$ in the above formula.
 * 
 * Optionally, you may implement:
 * 4. The method `DFimpl`, which computes the Jacobian of \f$ F \f$. If this 
 *    method is not implemented, a numerical approximation is used.
 *    If it is implemented, add `static constexpr bool useDFimpl = true`;
 *    otherwise, set `static constexpr bool useDFimpl = false`.
 * 5. The method `initialValueimpl`, which provides an initial value for  
 *    root-finding algorithms. If this method is not implemented, a random  
 *    vector with elements uniformly distributed between -1 and 1 is generated.
 *    If it is implemented, add `static constexpr bool useIniimpl = true`;
 *    otherwise, set `static constexpr bool useIniimpl = false`.
 * 
 * Example:
 * @code
 * class DerivedTargetFunction : public BaseTargetFunction<DerivedTargetFunction>
 * {
 *     friend BaseTargetFunction;
 * private:
 *     // Size of input and output vectors
 *     static constexpr size_t size = ...;
 *     // Whether DFimpl is defined
 *     static constexpr bool useDFimpl = ...;
 *     // Whether initialValueimpl is defined
 *     static constexpr bool useIniimpl = ...;
 *     // Implementation of the target function.
 *     template <typename EigenDerived>
 *     Eigen::Matrix<double, size, 1> 
 *     Fimpl(const Eigen::MatrixBase<EigenDerived>& z)
 *     {...}
 *     // Optional: Implementation of the Jacobian of the target function.
 *     Eigen::Matrix<double, size, size> 
 *     DFimpl(const Eigen::MatrixBase<EigenDerived>& z)
 *     {...}
 *     // Optional: Implementation of the initial value generator.
 *     Eigen::Matrix<double, size, 1> 
 *     initialValueimpl()
 *     {...}
 * };
 * @endcode
 */
template<class DerivedTargetFunction>
class BaseTargetFunction
{
public: // Main methods
	/**
	 * @brief Evaluates the target function F(z).
	 * @param z Input vector.
	 * @return F(z).
	 */
	template <typename EigenDerived>
	auto F(const Eigen::MatrixBase<EigenDerived>& z) const;

	/**
	 * @brief Evaluates the Jacobian of the target function F(z).
	 * @param z Input vector.
	 * @return Jacobian of F(z).
	 */
	template <typename EigenDerived>
	auto DF(const Eigen::MatrixBase<EigenDerived>& z) const;

	/**
	 * @brief Provides the initial value for root-finding algorithms.
	 * @return Initial value vector.
	 */
	auto initialValue() const;

	/**
	 * @brief Size of input/output vector of target function.
	 */
	static constexpr size_t size();

public: // Default implementations for optional methods
	/**
	 * @brief Numerical approximation of the Jacobian of F(z).
	 * @param z Input vector.
	 * @return Approximated Jacobian of F(z).
	 */
	template <typename EigenDerived>
	auto DFApprox(const Eigen::MatrixBase<EigenDerived>& z) const;

	/**
	 * @brief Generates a random initial value for root-finding algorithms.
	 * Each element is uniformly distributed between -1 and 1.
	 * @return Random initial value vector.
	 */
	auto defaultInitialValue() const;

public:
	/**
	 * @brief Checks if the input/output argument `z` has the correct dimensions.
	 */
	template <typename EigenDerived>
	void checkArgumentCorrectness(const Eigen::MatrixBase<EigenDerived>& z) const;

protected: // Hide potentially danger methods to avoid UB
    BaseTargetFunction() = default;
    BaseTargetFunction(const BaseTargetFunction<DerivedTargetFunction>&) = default;
    BaseTargetFunction(BaseTargetFunction<DerivedTargetFunction>&&) = default;
	BaseTargetFunction<DerivedTargetFunction>& operator=(
		const BaseTargetFunction<DerivedTargetFunction>&) = default;
    BaseTargetFunction<DerivedTargetFunction>& operator=(
		BaseTargetFunction<DerivedTargetFunction>&&) = default;
	~BaseTargetFunction() = default;
};
} //namespace rootfinder

#include "BaseTargetFunction.ipp"
#endif //BASE_TARGET_FUNCTION_H