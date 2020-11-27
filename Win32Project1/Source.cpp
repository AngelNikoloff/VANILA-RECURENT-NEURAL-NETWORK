// Author:		John McCullock
// Date:		10-15-05
// Description:	Elman Network Example 1.
// http://mnemstudio.org/neural-networks-elman.htm

#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <ctime>
#include <cstdlib>
#include <vector>

namespace VANIAL_RNN
{
	//=========================================================================================================================================== helper functions;
	namespace ANN_TINY_LIB
	{
		//=========================================================================================================================================== HELPERS;
		int getRandomNumber()
		{
			return int(6 * rand() / (RAND_MAX + 1.0));	             // Generate random value between 0 and 6.
		}
		double sigmoid(double val)
		{
			return (1.0 / (1.0 + exp(-val)));
		}
		double sigmoidDerivative(double val)
		{
			return (val * (1.0 - val));
		}
		//======================================================================================================================
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
		std::vector<double> dotVM(const std::vector<double> &vector, const std::vector<std::vector<double>> &matrix)
		{

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
		std::vector<std::vector<double>> deltaWeights_b(const double &learnRate, const std::vector<double> &v_layer_input, const std::vector<double> &v_layer_error)
		{
			int input_Neurons = v_layer_input.size(); // hid
			int output_Neurons = v_layer_error.size(); // out

			std::vector<std::vector<double>> result(input_Neurons + 1, std::vector<double>(output_Neurons, 0));

			for (int out = 0; out <= (output_Neurons - 1); out++)
			{
				for (int hid = 0; hid <= (input_Neurons - 1); hid++)
				{
					result[hid][out] += (learnRate * v_layer_input[hid] * v_layer_error[out]);
				}

				result[input_Neurons][out] += (learnRate * v_layer_error[out]); // Update the bias.
			}

			return result;
		}
		//=================
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
		template<typename T> std::vector<std::vector<T>> operator + (const std::vector<std::vector<T>> &lhs, const std::vector<std::vector<T>> &rhs)
		{
			typedef std::vector<T>::size_type size_type;
			if (lhs.size() != rhs.size()) throw std::length_error("vectors must be same size to multiplicate");
			if (lhs[0].size() != rhs[0].size()) throw std::length_error("vectors must be same size to multiplicate");

			int rr = lhs.size();
			int cc = lhs[0].size();

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
		//===========================================================================================================================================
		template<typename T> double sum(const std::vector<T> &_vector)
		{
			double sum = 0;

			for (unsigned int i = 0; i < _vector.size(); i++) sum += _vector[i];

			return sum;
		}
		template<typename T> double max_abs(const std::vector<T> &_vector)
		{
			double max = 0;

			for (unsigned int i = 0; i < _vector.size(); i++) if (abs(_vector[i]) > max) max = abs(_vector[i]);

			return max;
		}
		template<typename T> int getCharIndex(const std::vector<T> &_vector)
		{
			double max = 0;
			int index = 0;

			for (unsigned int i = 0; i < _vector.size(); i++) if (abs(_vector[i]) > max)
			{
				index = i;
				max = abs(_vector[i]);
			}
			return index;
		}
		//===========================================================================================================================================
	} using namespace ANN_TINY_LIB;
	//===========================================================================================================================================

	//===========================================================================================================================================
	const int maxTests = 10000;
	const int maxSamples = 4;

	const int inputNeurons = 6;
	const int hiddenNeurons = 3;
	const int contextNeurons = 3;
	const int outputNeurons = 6;

	const double learnRate = 0.2;    //Rho.
	const int trainingReps = 2000;  // 0=2000/0.596; // 4000=0.301;

	std::vector<double> beVector = { 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 };                   // beVector is the symbol used to start or end a sequence.
	//=========================================================================================================================================== //  0    1    2    3    4    5
	std::vector<std::vector<double>> sampleInput = {
		{ 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 },
		{ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 },
		{ 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 }
	};
	//=========================================================================================================================================== Unit errors.
	std::vector<double> err_o(outputNeurons, 0);
	std::vector<double> err_h(hiddenNeurons, 0);
	//=========================================================================================================================================== Activations.
	std::vector<double> v_inputs(inputNeurons, 0);
	std::vector<double> v_hidden(hiddenNeurons, 0);
	std::vector<double> v_target(outputNeurons, 0);
	std::vector<double> v_output(outputNeurons, 0);
	std::vector<double> v_context(contextNeurons, 0);
	//===========================================================================================================================================
	std::vector<std::vector<double>> wih(inputNeurons + 1, std::vector<double>(hiddenNeurons, 0));      // Input to v_hidden weights (with biases).
	std::vector<std::vector<double>> wch(contextNeurons + 1, std::vector<double>(hiddenNeurons, 0));// v_context to v_hidden weights (with biases).
	std::vector<std::vector<double>> who(hiddenNeurons + 1, std::vector<double>(outputNeurons, 0));    // v_hidden to Output weights (with biases).
	//===========================================================================================================================================



	//===========================================================================================================================================
	void assignPseudoRandomWeights()
	{
		srand((unsigned)3);           // pseudo random;

		//srand((unsigned)time(0));   // random - Seed random number generator with system time.

		for (int inp = 0; inp <= inputNeurons; inp++) {
			for (int hid = 0; hid <= (hiddenNeurons - 1); hid++) {
				wih[inp][hid] = -0.5 + double(rand() / (RAND_MAX + 1.0));                   // Assign a random weight value between -0.5 and 0.5;
			}
		}

		for (int con = 0; con <= contextNeurons; con++) {
			for (int hid = 0; hid <= (hiddenNeurons - 1); hid++) {
				wch[con][hid] = -0.5 + double(rand() / (RAND_MAX + 1.0));                   // Assign a random weight value between -0.5 and 0.5;
			}
		}

		for (int hid = 0; hid <= hiddenNeurons; hid++) {
			for (int out = 0; out <= (outputNeurons - 1); out++) {
				who[hid][out] = -0.5 + double(rand() / (RAND_MAX + 1.0));                   // Assign a random weight value between -0.5 and 0.5;
			}
		}
	}
	//===========================================================================================================================================
	void feedForward()
	{
		v_hidden = dotVM_b(v_inputs, wih) + dotVM_b(v_context, wch);              // Calculate input and v_context connections to v_hidden layer.
		v_hidden = sigm(v_hidden);

		v_output = dotVM_b(v_hidden, who);                                                            //  Calculate the v_hidden to output layer.
		v_output = sigm(v_output);

		v_context = v_hidden;                                                                 // Copy outputs of the v_hidden to v_context layer.
	}
	void backPropagate()
	{
		err_o = (v_target - v_output) * sigmoidDerivative(v_output);                // Calculate the output layer error (step 3 for output cell).              // gradient = delta_err * derivativ;                                 
		err_h = dotVM(err_o, who)     * sigmoidDerivative(v_hidden);                // Calculate the hidden layer error (step 3 for hidden cell).

		who = who + deltaWeights_b(learnRate, v_hidden, err_o);                              // Update the weights for the output layer (step 4).  
		wih = wih + deltaWeights_b(learnRate, v_inputs, err_h);                              // Update the weights for the hidden layer (step 4).
	}
	//===========================================================================================================================================


	//===========================================================================================================================================
	void train_Network()
	{
		assignPseudoRandomWeights();

		int sample = 0;

		int oki_error = 0.0;
		double Mean_squared_error = 0;
		std::vector<double> d_error_o(outputNeurons, 0);

		for (int epoch = 1; epoch <= trainingReps; epoch++)
		{
			//============================================================= // After the samples are entered into the input units, the sample are then offset by one and entered into v_target-output units for later comparison.
			if (sample == 0)   v_inputs = beVector;
			else			   v_inputs = sampleInput[sample - 1];


			if (sample == maxSamples - 1) 	v_target = beVector;
			else			                v_target = sampleInput[sample];
			//=============================================================

			feedForward();

			//============================================================= CALC ERROR;
			if (sample == 0)
			{
				oki_error = 0.0;
				Mean_squared_error = 0.0;
			}

			if (sample != 0)
			{
				d_error_o = v_target - v_output;
				d_error_o = d_error_o * d_error_o;
				Mean_squared_error += sum(d_error_o);

				if(getCharIndex(v_output) != getCharIndex(v_target))	oki_error += 1;
			}

			if (sample == maxSamples - 1)
			{
				Mean_squared_error = Mean_squared_error / maxSamples;

				std::cout << "epoch = " << epoch << "    Mean_squared_error = " << Mean_squared_error << "    Number of urong predictions in one epoch = " << oki_error << "    "; std::cout << std::endl;
			}

			//=============================================================

			backPropagate();

			//=============================================================
			sample += 1;
			if (sample == maxSamples) sample = 0;
			//=============================================================
		}




	}
	void test_Network()
	{
		// we expect = 0352 0352 0352 0352 0352 0352 

		std::cout << std::endl << "************ Start test 1 ************" << std::endl << std::endl;

		v_inputs = beVector;             // Enter Beginning string.

		int O = getCharIndex(v_inputs);

		std::cout << "test: " << 0 << " start sumbol = " << O << std::endl;

		for (int test = 1; test <= 21; test++)
		{

			feedForward();

			int predicted_char = getCharIndex(v_output);

			if (predicted_char == 0) std::cout << std::endl;

			std::cout << "test: " << test << "    predicted_char = " << predicted_char << std::endl;

			v_inputs = v_output;
		}

	}
	//===========================================================================================================================================
};

//===============================================================================================================================================
int main()
{
	using namespace VANIAL_RNN;

	std::cout << std::fixed << std::setprecision(3) << std::endl;                                                   // Format all the output.

	train_Network();
	test_Network();

	system("pause");

	return 0;
}
//===============================================================================================================================================




// todo - noise when training;



