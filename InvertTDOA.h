#ifndef INVERT_TDOA_H
#define INVERT_TDOA_H

#include "RootFinder/RootFinder.h"
#include "Eigen/Dense"
#include <random>

/**
 * @brief A class describing a 2D Invert TDOA function and its Jacobian.
 * 
 * Invert TDOA function represents a function 
 * \f$ F_{x^1, x^2, x^3} : (y^1, y^2, y^3) \rightarrow \mathbb{R}^6 \f$
 * where 
 * \f$ y^1, y^2, y^3 \f$ are coordinates of three signal receivers; and 
 * \f$ x^1, x^2, x^3 \f$ are coordinates of three signal sources, known inner 
 * parameters of the function.
 * 
 * Denote \f$ \delta^{ijk} := \| y^i - x^k \| - \| y^j - x^k \| \f$. Then:
 * \f[
 * F_{x^1, x^2, x^3}(y^1, y^2, y^3) = 
 * (\delta^{121}, \delta^{231}, \delta^{122}, 
 *  \delta^{232}, \delta^{123}, \delta^{233})
 * \f]
 */
class InvertTDOA : public rootfinder::BaseTargetFunction<InvertTDOA>
{
	friend rootfinder::BaseTargetFunction<InvertTDOA>;
public:
	/**
	 * @brief InvertTDOA constructor.
	 * 
	 * @param x1 2d vector representing the coordinates of the first signal source.
	 * @param x2 2d vector representing the coordinates of the second signal source.
	 * @param x3 2d vector representing the coordinates of the third signal source.
	 */
	InvertTDOA(const Eigen::Vector2d& x1, const Eigen::Vector2d& x2, 
		const Eigen::Vector2d& x3);

public: // Required in BaseTargetFunction fields.
	static constexpr size_t size = 6;
    static constexpr bool useDFimpl = true;
    static constexpr bool useIniimpl = true;

private: // Required in BaseTargetFunction methods.

	/**
	 * @brief Invert TDOA function
	 * \f$ F: \mathbb{R}^6 \rightarrow \mathbb{R}^6 \f$
	 * 
	 * @param y Vector of size 6 \f$ (y_0, \dots, y_5) \f$ containing 
     * (potential) coordinates of receivers: 
	 * \f$ y^1 = (y_0, y_1) \f$ is the first receiver coordinates, 
	 * \f$ y^2 = (y_2, y_3) \f$ -- the second, 
	 * \f$ y^3 = (y_4, y_5) \f$ -- the third.
	 * @return Invert TDOA function value. 
	 */
	template <typename EigenDerived>
	Eigen::Matrix<double, size, 1> 
	Fimpl(const Eigen::MatrixBase<EigenDerived>& z) const;

	/**
	 * @brief Invert TDOA Jacobian.
	 * 
	 * @param y Vector of size 6 \f$ (y_0, \dots, y_5) \f$ containing (potential) 
	 * receivers coordinates: 
	 * \f$ y^1 = (y_0, y_1) \f$ is the first receiver coordinates, 
	 * \f$ y^2 = (y_2, y_3) \f$ -- the second, 
	 * \f$ y^3 = (y_4, y_5) \f$ -- the third.
	 * @return Invert TDOA Jacobian. 
	 */
	template <typename EigenDerived>
	Eigen::Matrix<double, size, size> 
	DFimpl(const Eigen::MatrixBase<EigenDerived>& z) const;

	/**
	 * @brief Initial value for invert TDOA. It is defined as a multivariate
     * distribution with mean in the baricenter of signal sources coordinates 
     * and  covariance matrix equal to \f$ r Id \f$, where \f$ r \f$ is a 
     * maximal difference between sources and their barycenter and \f$ Id \f$
     * is an identity matrix.
     * 
     * @return Initial value vector.
	 */
    Eigen::Matrix<double, size, 1> initialValueimpl() const;

private: // inner parameters of the function
	/// @brief Three signal sources with known parameters.
	Eigen::Vector2d x1_, x2_, x3_;

};

#include "InvertTDOA.ipp"
#endif // INVERT_TDOA_H