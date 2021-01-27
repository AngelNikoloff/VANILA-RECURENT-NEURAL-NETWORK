#pragma once

#include <vector>
#include <iostream>

namespace ANN_TINY_LIB_100
{
	//=========================================================================================================================================== HELPERS;
	int _getRandomNumber()
	{
		return int(6 * rand() / (RAND_MAX + 1.0));	             // Generate random value between 0 and 6.
	}
	//=================================
	double sigmoid(double val)
	{
		return (1.0 / (1.0 + exp(-val)));
	}
	double sigmoidDerivative(double val)
	{
		return (val * (1.0 - val));
	}
	std::vector<double> sigm(std::vector<double> vector)
	{
		//std::vector<double> result(vector.size(), 0);

		for (int out = 0; out <= (vector.size() - 1); out++)
		{
			vector[out] = sigmoid(vector[out]);
		}

		return vector;
	}
	std::vector<double> sigmoidDerivative(std::vector<double> vector)
	{
		//std::vector<double> result(vector.size(), 0);

		for (int out = 0; out <= (vector.size() - 1); out++)
		{
			vector[out] = sigmoidDerivative(vector[out]);
		}

		return vector;
	}
	template<typename T> double sum(const std::vector<T> &_vector)
	{
		double sum = 0;

		for (unsigned int i = 0; i < _vector.size(); i++) sum += _vector[i];

		return sum;
	}
	//==========================================================================================================================================


	//========================================================================================================================================== with included bias;
	std::vector<double> dotVM_b(const std::vector<double> &vector, const std::vector<std::vector<double>> &matrix)
	{
		int hiddenNeurons = matrix.size() - 1;
		int outputNeurons = matrix[0].size();

		std::vector<double> result(matrix[0].size(), 0);

		double sum;

		for (int out = 0; out <= (outputNeurons - 1); out++)
		{
			sum = 0.0;
			for (int hid = 0; hid <= (hiddenNeurons - 1); hid++)
			{
				sum += vector[hid] * matrix[hid][out];
			}

			sum += matrix[hiddenNeurons][out];     // Add in bias.

			result[out] = sum;
		}

		return result;
	}
	std::vector<std::vector<double>> deltaWeights_b(const std::vector<double> &v_layer_input, const std::vector<double> &v_layer_error)
	{
	// ne += !!!

		int input_Neurons = v_layer_input.size(); // hid
		int output_Neurons = v_layer_error.size(); // out

		std::vector<std::vector<double>> result(input_Neurons + 1, std::vector<double>(output_Neurons, 0));
		//std::vector<std::vector<double>> result(output_Neurons + 1, std::vector<double>(input_Neurons, 0));


		for (int out = 0; out <= (output_Neurons - 1); out++)
		{
			for (int hid = 0; hid <= (input_Neurons - 1); hid++)
			{
				result[hid][out] += (v_layer_input[hid] * v_layer_error[out]); /// samo = tribva
			}

			result[input_Neurons][out] += (v_layer_error[out]); // Update the bias.
		}

		return result;
	}
	//==========================================================================================================================================
	std::vector<double> dotVM_(const std::vector<double> &vector, const std::vector<std::vector<double>> &matrix)
	{
		int hiddenNeurons = matrix.size() - 1;
		int outputNeurons = matrix[0].size();

		std::vector<double> result(matrix[0].size(), 0);

		double sum;

		for (int out = 0; out <= (outputNeurons - 1); out++)
		{
			sum = 0.0;
			for (int hid = 0; hid <= (hiddenNeurons - 1); hid++)
			{
				sum += vector[hid] * matrix[hid][out];
			}

			//sum += matrix[hiddenNeurons][out];     // Add in bias.

			result[out] = sum;
		}

		return result;
	}
	std::vector<std::vector<double>> deltaWeights(const std::vector<double> &v_layer_input, const std::vector<double> &v_layer_error)
	{
		int input_Neurons = v_layer_input.size(); // hid
		int output_Neurons = v_layer_error.size(); // out

		std::vector<std::vector<double>> result(input_Neurons + 1, std::vector<double>(output_Neurons, 0));
		//std::vector<std::vector<double>> result(output_Neurons + 1, std::vector<double>(input_Neurons, 0));

		//std::cout << "aaa output_Neurons = " << output_Neurons << "    input_Neurons = " << input_Neurons << std::endl;

		for (int out = 0; out <= (output_Neurons - 1); out++)
		{
			for (int hid = 0; hid <= (input_Neurons - 1); hid++)
			{
				//std::cout << "bbb out = " << out << "    hid = " << hid << std::endl;

				result[hid][out] += (v_layer_input[hid] * v_layer_error[out]);
			}

			//result[input_Neurons][out] += (v_layer_error[out]); // Update the bias.
		}

		return result;
	}
	//========================================================================================================================================== // for BP
	std::vector<double> dotVM(const std::vector<double> &vector, const std::vector<std::vector<double>> &matrix)
	{
		/*
		int hiddenNeurons = matrix.size() - 1;
		int outputNeurons = matrix[0].size();

		std::vector<double> result(matrix[0].size(), 0);

		double sum;

		for (int out = 0; out <= (outputNeurons - 1); out++)
		{
		sum = 0.0;
		for (int hid = 0; hid <= (hiddenNeurons - 1); hid++)
		{
		sum += vector[hid] * matrix[hid][out];
		}

		//sum += matrix[hiddenNeurons][out];     // Add in bias.

		result[out] = sum;
		}
		*/
		//***************************************************************************
		int hiddenNeurons = matrix.size() - 1;
		int outputNeurons = matrix[0].size();

		std::vector<double> result(hiddenNeurons, 0);

		for (int hid = 0; hid <= (hiddenNeurons - 1); hid++)
		{
			//err_h[hid] = 0.0;
			for (int out = 0; out <= (outputNeurons - 1); out++)
			{
				result[hid] += vector[out] * matrix[hid][out];
			}
			//err_h[hid] *= sigmoidDerivative(v_hidden[hid]);
		}
		//***************************************************************************

		return result;
	}
	//==========================================================================================================================================



	//========================================================================================================================================== vector operations;
	template<typename T> std::vector<T> operator + (std::vector<T> lhs, const std::vector<T> &rhs)
	{
		typedef std::vector<T>::size_type size_type;

		if (lhs.size() != rhs.size()) throw std::length_error("vectors must be same size to add");

		for (size_type i = 0; i < lhs.size(); i++)	lhs[i] += rhs[i];

		return lhs;
	}
	template<typename T> std::vector<T> operator - (std::vector<T> lhs, const std::vector<T> &rhs)
	{
		typedef std::vector<T>::size_type size_type;

		if (lhs.size() != rhs.size()) throw std::length_error("vectors must be same size to add");

		for (size_type i = 0; i < lhs.size(); i++)	lhs[i] -= rhs[i];

		return lhs;
	}
	template<typename T> std::vector<T> operator * (std::vector<T> lhs, const std::vector<T> &rhs)
	{
		typedef std::vector<T>::size_type size_type;

		if (lhs.size() != rhs.size()) throw std::length_error("vectors must be same size to multiplicate");

		for (size_type i = 0; i < lhs.size(); i++)	lhs[i] *= rhs[i];

		return lhs;
	}
	template<typename T> std::vector<T> operator * (T lhs, std::vector<T> rhs)
	{
		typedef std::vector<T>::size_type size_type;


		for (size_type i = 0; i < rhs.size(); i++)	rhs[i] = rhs[i] * lhs;

		return rhs;
	}
	//=====================================================================================================
	template<typename T> std::vector<std::vector<T>> operator + (const std::vector<std::vector<T>> &lhs, const std::vector<std::vector<T>> &rhs)
	{
		typedef std::vector<T>::size_type size_type;
		if (lhs.size() != rhs.size()) throw std::length_error("vectors must be same size to multiplicate");
		if (lhs[0].size() != rhs[0].size()) throw std::length_error("vectors must be same size to multiplicate");

		int rr = lhs.size();
		int cc = lhs[0].size();

		//for (size_type i = 0; i < lhs.size(); i++)	lhs[i] *= rhs[i];

		std::vector<std::vector<T>> result(rr, std::vector<double>(cc, 0));

		for (int r = 0; r < rr; r++)
		{
			for (int c = 0; c < cc; c++)
			{
				result[r][c] = lhs[r][c] + rhs[r][c];
			}
		}

		return result;
	}
	template<typename T> std::vector<std::vector<T>> operator * (T lhs, const std::vector<std::vector<T>> rhs)
	{
		typedef std::vector<T>::size_type size_type;

		int rr = rhs.size();
		int cc = rhs[0].size();

		//for (size_type i = 0; i < lhs.size(); i++)	lhs[i] *= rhs[i];

		std::vector<std::vector<T>> result(rr, std::vector<T>(cc, 0));

		for (int r = 0; r < rr; r++)
		{
			for (int c = 0; c < cc; c++)
			{
				result[r][c] = lhs * rhs[r][c];

			}
		}

		return result;
	}
	//===========================================================================================================================================


	//===========================================================================================================================================


	//===========================================================================================================================================
	template<typename T> int getCharIndex(const std::vector<T> &_vector)
	{
		double max_num = 0;
		int index = 0;

		for (unsigned int i = 0; i < _vector.size(); i++)
			if (abs(_vector[i]) > max_num)
			{
				index = i;
				max_num = abs(_vector[i]);
			}
		return index;
	}
	std::vector<double> charToVector(const char &c)
	{
		int m_input_size = 128;

		std::vector<double> one_hot_vector(m_input_size,0);
		one_hot_vector[(int)c] = 1;

		return one_hot_vector;
	}
	char vectorToChar(const std::vector<double> &_vector)
	{
		double max_num = 0;
		int index = 0;

		for (unsigned int i = 0; i < _vector.size(); i++)
			if (abs(_vector[i]) > max_num)
			{
				index = i;
				max_num = abs(_vector[i]);
			}
		return (char)index;

	}
	//===========================================================================================================================================
} using namespace ANN_TINY_LIB_100;












/*


	template<typename T> double max_abs(const std::vector<T> &_vector)
	{
		double max = 0;

		for (unsigned int i = 0; i < _vector.size(); i++) 
			if (abs(_vector[i]) > max) max = abs(_vector[i]);

		return max;
	}



*/













