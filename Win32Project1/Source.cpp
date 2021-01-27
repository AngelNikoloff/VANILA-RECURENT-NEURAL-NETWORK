//===========================================================================================================================================
#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <ctime>
#include <cstdlib>
#include <vector>
//===========================================================================================================================================

// ***** Simple Recurrent Network (Elman, 1990) ***** 
// Simple  Recurrent  Networks  (SRNs)  can  learn  medium-rangedependencies but have difficulty learning long range dependencies

//===========================================================================================================================================
namespace DATA
{
    #include "ANN_LIB.h"

	//===========================================================================================================================================
	const double learnRate = 0.2;
	const double momentum = 0.5;
	const int epochs = 500000;
	//===========================================================================================================================================


	//===========================================================================================================================================
	std::vector<double> emptyVector(128, 0);
	std::vector<double> randomVector(128, 0);
	//===========================================================================================================================================


	//=========================================================================================================================================== 128 is number of suported sumbols;
	const int inputNeurons = 128;
	const int hiddenNeurons = 128;
	const int contextNeurons = 128;
	const int outputNeurons = 128;
	//=========================================================================================================================================== Unit errors.
	std::vector<double> err_o(outputNeurons, 0);
	std::vector<double> err_h(hiddenNeurons, 0);
	//=========================================================================================================================================== Activations.
	std::vector<double> v_inputs(inputNeurons, 0);
	std::vector<double> v_hidden(hiddenNeurons, 0);
	std::vector<double> v_target(outputNeurons, 0);
	std::vector<double> v_output(outputNeurons, 0);
	std::vector<double> v_context(contextNeurons, 0);
	//===============================================
	std::vector<std::vector<double>> wih(inputNeurons + 1, std::vector<double>(hiddenNeurons, 0));      // Input to v_hidden weights (with biases).
	std::vector<std::vector<double>> wch(contextNeurons + 1, std::vector<double>(hiddenNeurons, 0));// v_context to v_hidden weights (with biases).
	std::vector<std::vector<double>> who(hiddenNeurons + 1, std::vector<double>(outputNeurons, 0));    // v_hidden to Output weights (with biases).
	//=============================================== // Bias Vectors
	std::vector<double> w_h_b(hiddenNeurons, 0);
	std::vector<double> w_o_b(outputNeurons, 0);
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
};
//===========================================================================================================================================



//=========================================================================================================================================== VARIANTS - CHOICE ONE IN THE BEGINING OF  "namespace RECURENT_NEURAL_NETWORK" !!!
namespace SIMPLE_RNN
{
	using namespace DATA;
	//======================================================================================================================================= VANILA;
	void feedForward()
	{
		v_hidden = dotVM_(v_inputs, wih) + dotVM_(v_hidden, wch) + w_h_b;       // Calculate input and v_context connections to v_hidden layer.
		v_hidden = sigm(v_hidden);
                                                  
		v_output = dotVM_(v_hidden, who) + w_o_b;                                                    //  Calculate the v_hidden to output layer.
		v_output = sigm(v_output);                                                            // Copy outputs of the v_hidden to v_context layer.
	}
	void backPropagate()
	{

		err_o = (v_target - v_output) * sigmoidDerivative(v_output);                // Calculate the output layer error (step 3 for output cell).              // gradient = delta_err * derivativ;                                 
		err_h = dotVM(err_o, who)     * sigmoidDerivative(v_hidden);                // Calculate the hidden layer error (step 3 for hidden cell).

		who = who + learnRate * deltaWeights(v_hidden, err_o);                             // Update the weights for the output layer (step 4). 
		wih = wih + learnRate * deltaWeights(v_inputs, err_h);                    // Update the weights for the input to hidden layer (step 4).
		wch = wch + learnRate * deltaWeights(v_context, err_h);

		w_o_b = w_o_b + learnRate * err_o;
		w_h_b = w_h_b + learnRate * err_h;

		v_context = v_hidden;
	}
	//=======================================================================================================================================
};

namespace SIMPLE_RNN_WITH_MOMENTUM
{
	using namespace DATA;
	// momentum https://machinelearningmastery.com/learning-rate-for-deep-learning-neural-networks/
	//=======================================================================================================================================	MOMENTUM;
	std::vector<std::vector<double>> d_wih(inputNeurons + 1, std::vector<double>(hiddenNeurons, 0));      // Input to v_hidden weights (with biases).
	std::vector<std::vector<double>> d_wch(contextNeurons + 1, std::vector<double>(hiddenNeurons, 0));// v_context to v_hidden weights (with biases).
	std::vector<std::vector<double>> d_who(hiddenNeurons + 1, std::vector<double>(outputNeurons, 0));    // v_hidden to Output weights (with biases).
	std::vector<double> d_w_o_b(outputNeurons, 0);
	std::vector<double> d_w_h_b(hiddenNeurons, 0);
	//=======================================================================================================================================
	void feedForward()
	{
		v_hidden = dotVM_(v_inputs, wih) + dotVM_(v_hidden, wch) + w_h_b;
		v_hidden = sigm(v_hidden);

		v_output = dotVM_(v_hidden, who) + w_o_b;                                                 //  Calculate the v_hidden to output layer.
		v_output = sigm(v_output);                                                        // Copy outputs of the v_hidden to v_context layer.
	}
	void backPropagate()
	{
		err_o = (v_target - v_output) * sigmoidDerivative(v_output);             // Calculate the output layer error (step 3 for output cell).              // gradient = delta_err * derivativ;                                 
		err_h = dotVM(err_o, who)     * sigmoidDerivative(v_hidden);             // Calculate the hidden layer error (step 3 for hidden cell).

		d_who = momentum * d_who + learnRate * deltaWeights(v_hidden, err_o);             // Update the weights for the output layer (step 4). 
		d_wih = momentum * d_wih + learnRate * deltaWeights(v_inputs, err_h);    // Update the weights for the input to hidden layer (step 4).
		d_wch = momentum * d_wch + learnRate * deltaWeights(v_context, err_h);

		who = who + d_who;                                                               // Update the weights for the output layer (step 4). 
		wih = wih + d_wih;                                                      // Update the weights for the input to hidden layer (step 4).
		wch = wch + d_wch;

		d_w_o_b = momentum * d_w_o_b + learnRate * err_o;
		d_w_h_b = momentum * d_w_h_b + learnRate * err_h;

		w_o_b = w_o_b + d_w_o_b;
		w_h_b = w_h_b + d_w_h_b;

		v_context = v_hidden;
	}
	//=======================================================================================================================================
};
//===========================================================================================================================================



//===========================================================================================================================================
namespace RECURENT_NEURAL_NETWORK
{
	//using namespace SIMPLE_RNN;
	using namespace SIMPLE_RNN_WITH_MOMENTUM;
	//============================================================================== memorization

	std::string txt = ".Simple Recurrent Network.";

	//===========================================================================================================================================
	void train_Network()
	{
		assignPseudoRandomWeights();

		int rpos = 0;
		int rchar = 0;
		int counter = 0;
		int oki_error = 0.0;
		double Mean_squared_error = 0;
		std::vector<double> d_error_o(outputNeurons, 0);

		//int sample = 0;
		int maxSamples = txt.size();

		for (int epoch = 1; epoch <= epochs; epoch++)
		{
			for (int sample = 0; sample < maxSamples; sample++)
			{
				//============================================================= // After the samples are entered into the input units, the sample are then offset by one and entered into v_target-output units for later comparison.
				v_inputs = charToVector(txt[sample]);
				v_target = charToVector(txt[sample + 1]);
				//============================================================= ADD NOISE IF YOU WANT;

				if (false)
				{
					//  NOISE;

					rpos = rand() % 5 + 1;
					rchar = -1;

					if (rpos == 5)
					{
						randomVector = emptyVector;

						//int noises = txt.size()/3;

						//for (int i = 1; i <= 5; i++)
						{}//}
						rchar = rand() % 127 + 0;
						randomVector[rchar] = 1;

						v_inputs = randomVector;
					}
				}
				
				feedForward();

				if(true)
				{
					//  CALC ERROR;

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

						if (getCharIndex(v_output) != getCharIndex(v_target))	oki_error += 1;
					}

					if (sample == maxSamples - 1)
					{
						counter++;

						Mean_squared_error = Mean_squared_error / maxSamples;

						if (counter > 10)
						{
							std::cout << "epoch = " << epoch << "  char: " << txt[sample] << "    Mean_squared_error = " << Mean_squared_error << "    Number of urong predictions in one epoch = " << oki_error << "     rpos: " << rpos << "  rchar:  " << rchar;   std::cout << std::endl;
							counter = 0;
						}
						//if (Mean_squared_error <= 0.003) break;
						if (Mean_squared_error <= 0.003) return;
					}
                }
				
				backPropagate();
				//=============================================================
			}
		}

		std::cout << "    Mean_squared_error = " << Mean_squared_error << std::endl;

	}
	void test_Network()
	{
		std::cout << std::endl << "************ Start test 1 ************" << std::endl << std::endl;

		int maxTests = txt.size() * 5;
		std::cout << "txt.size() = " << txt.size() << std::endl << std::endl;


		//v_inputs = beVector;             // Enter Beginning string.
		v_inputs = charToVector(txt[0]);


		//int start = getCharIndex(v_inputs);
		char start = vectorToChar(v_inputs);
		//std::cout << "test: " << 1 << "      start sumbol = " << start << std::endl;
		std::cout << start;




		for (int test = 1; test < maxTests; test++)
		{

			feedForward();

			//int predicted_char = getCharIndex(v_output);
			int index_char = getCharIndex(v_output);
			char predicted_char = vectorToChar(v_output);

			//if (predicted_char == 0) std::cout << std::endl;
			if (predicted_char == txt[0]) std::cout << std::endl;

			//std::cout << "test: " << test + 1 << "    predicted_char = " << predicted_char << std::endl;

			if (index_char > 31)	std::cout << predicted_char;


			v_inputs = v_output;
		}
		std::cout << std::endl;
	}
	//===========================================================================================================================================
};
//===========================================================================================================================================



//===========================================================================================================================================
int main()
{
	std::cout << std::fixed << std::setprecision(3) << std::endl;                                                       // Format all the output.

	RECURENT_NEURAL_NETWORK::train_Network();
	RECURENT_NEURAL_NETWORK::test_Network();

	system("pause");

	return 0;
}
//===========================================================================================================================================




