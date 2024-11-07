#include "BaseTargetFunction.h"
#include "InternalHeaderCheck.h"

namespace rootfinder
{

template <class DerivedTargetFunction>
constexpr size_t BaseTargetFunction<DerivedTargetFunction>::size() 
{
    return DerivedTargetFunction::size;
}


template <class DerivedTargetFunction>
template <typename EigenDerived>
void BaseTargetFunction<DerivedTargetFunction>::checkArgumentCorrectness(
    const Eigen::MatrixBase<EigenDerived>& z) const
{
    static_assert(EigenDerived::RowsAtCompileTime == size() || 
        EigenDerived::RowsAtCompileTime == -1,
        "Incorrect number of rows in input `z`");
    if constexpr (EigenDerived::RowsAtCompileTime == -1)
        assert(z.rows() == size() && "Incorrect number of rows in input `z`");
    static_assert(EigenDerived::ColsAtCompileTime == 1 ||
        EigenDerived::ColsAtCompileTime == -1,
        "Input `z` must be a column vector");
    if constexpr (EigenDerived::ColsAtCompileTime == -1)
        assert(z.cols() == 1 && "Input `z` must be a column vector");
}


template <class DerivedTargetFunction>
template <typename EigenDerived>
auto BaseTargetFunction<DerivedTargetFunction>::F(
    const Eigen::MatrixBase<EigenDerived>& z) const
{
    checkArgumentCorrectness(z);
    return static_cast<const DerivedTargetFunction*>(this)->Fimpl(z);
}


template <class DerivedTargetFunction>
template <typename EigenDerived>
auto BaseTargetFunction<DerivedTargetFunction>::DF(const Eigen::MatrixBase<EigenDerived>& z) const
{
    checkArgumentCorrectness(z);
    if constexpr (DerivedTargetFunction::useDFimpl)
        return static_cast<const DerivedTargetFunction*>(this)->DFimpl(z);
    else
        return DFApprox(z);
}


template <class DerivedTargetFunction>
auto BaseTargetFunction<DerivedTargetFunction>::initialValue() const
{
    if constexpr (DerivedTargetFunction::useIniimpl)
        return static_cast<const DerivedTargetFunction*>(this)->initialValueimpl();
    else
        return defaultInitialValue();
}


template <class DerivedTargetFunction>
template <typename EigenDerived>
auto BaseTargetFunction<DerivedTargetFunction>::DFApprox(
    const Eigen::MatrixBase<EigenDerived>& z) const
{
    constexpr double epsilon = 1e-5;
    checkArgumentCorrectness(z);
    Eigen::Matrix<double, size(), size()> result;
    for (size_t col = 0; col < size(); ++col)
    {
        Eigen::Matrix<double, size(), 1> delta = 
            Eigen::Matrix<double, size(), 1>::Zero();
        delta(col, 0) = epsilon;
        result.col(col) = (F(z + delta) - F(z)) / epsilon;
    }
    return result;
}


template <class DerivedTargetFunction>
auto BaseTargetFunction<DerivedTargetFunction>::defaultInitialValue() const
{
    return Eigen::Matrix<double, size(), 1>::Random(); 
}
} // namespace rootfinder