#include "InvertTDOA.h"
#include "InvertTDOAsolver.h"
#include <stdexcept>
#include <iostream> // remove it

std::array<Eigen::Vector2d, 3> invertTDOAsolver(
	const std::array<Eigen::Vector2d, 3> &sourceCoordinates,
	const Eigen::Matrix3d& pddMatrix)
{
	const auto& [x1, x2, x3] = sourceCoordinates;
	Eigen::Matrix<double, 6, 1> rhs;
	rhs << pddMatrix(0,0), pddMatrix(0,2),
		   pddMatrix(1,0), pddMatrix(1,2),
		   pddMatrix(2,0), pddMatrix(2,2);

	InvertTDOA itdoa(x1,x2,x3);
	rootfinder::LevenbergMarquardtRootFinder<InvertTDOA> rf(itdoa, rhs);

	// Try to run root finder 10 times. If all running return incorrect
	// root, throw a error.
	for(size_t repeat = 0; repeat < 100; ++repeat)
	{
		while(rf.step()); //std::cout << rf.getResidual() << " "; std::cout << "\n\n\n----------------\n";
		if(rf.getResidual() < 1e-10) {break;}
		rf.reset();
	}

	if(rf.getResidual() > 1e-10)
	{
		throw std::runtime_error("Failed to find a satisfactory root " 
			"approximation.");
	}

	std::array<Eigen::Vector2d, 3> result;
	result[0] = rf.getCurrentZ().block(0,0,2,1);
	result[1] = rf.getCurrentZ().block(2,0,2,1);
	result[2] = rf.getCurrentZ().block(4,0,2,1);
	return result;
}