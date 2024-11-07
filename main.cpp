#include "InvertTDOAsolver.h"
#include "Eigen/Dense"
#include <iostream>
#include <array>


Eigen::Matrix3d getPddMatrix(
	const Eigen::Vector2d& x1, const Eigen::Vector2d& x2, const Eigen::Vector2d& x3,
	const Eigen::Vector2d& y1, const Eigen::Vector2d& y2, const Eigen::Vector2d& y3)
{
	Eigen::Matrix3d pddMatrix;
	pddMatrix(0,0) = (y1 - x1).norm() - (y2 - x1).norm();
	pddMatrix(0,1) = (y1 - x1).norm() - (y3 - x1).norm();
	pddMatrix(0,2) = (y2 - x1).norm() - (y3 - x1).norm();
	pddMatrix(1,0) = (y1 - x2).norm() - (y2 - x2).norm();
	pddMatrix(1,1) = (y1 - x2).norm() - (y3 - x2).norm();
	pddMatrix(1,2) = (y2 - x2).norm() - (y3 - x2).norm();
	pddMatrix(2,0) = (y1 - x3).norm() - (y2 - x3).norm();
	pddMatrix(2,1) = (y1 - x3).norm() - (y3 - x3).norm();
	pddMatrix(2,2) = (y2 - x3).norm() - (y3 - x3).norm();
	return pddMatrix;
}


int main()
{
	// Generate input parameters.
	const Eigen::Vector2d x1 = Eigen::Vector2d::Random();
	const Eigen::Vector2d x2 = Eigen::Vector2d::Random(); 
	const Eigen::Vector2d x3 = Eigen::Vector2d::Random();
	const Eigen::Vector2d y1Real = 100*Eigen::Vector2d::Random();
	const Eigen::Vector2d y2Real = 100*Eigen::Vector2d::Random(); 
	const Eigen::Vector2d y3Real = 100*Eigen::Vector2d::Random();

	Eigen::Matrix3d pddMatrixReal = getPddMatrix(x1,x2,x3,y1Real,y2Real,y3Real);

	try 
	{
		const auto [y1Approx, y2Approx, y3Approx] = invertTDOAsolver(
			std::to_array<Eigen::Vector2d>({x1,x2,x3}), pddMatrixReal);
		Eigen::Matrix3d pddMatrixApprox = 
			getPddMatrix(x1,x2,x3,y1Approx,y2Approx,y3Approx);

		std::cout << "Obtained y:\n";
		std::cout << "y1 = " << y1Approx.transpose() << std::endl;
		std::cout << "y2 = " << y2Approx.transpose() << std::endl;
		std::cout << "y3 = " << y3Approx.transpose() << std::endl;
		std::cout << "--------------------------" << std::endl;
		std::cout << "Accuracy (difference between real pddMatrix and "
			"obtained pddMatrix):\n";
		std::cout << pddMatrixApprox - pddMatrixReal << std::endl;

		return EXIT_SUCCESS;;
	}
	catch (const std::runtime_error& e) 
	{
		std::cerr << "Error: " << e.what() << std::endl;
		return EXIT_FAILURE;
	}
}