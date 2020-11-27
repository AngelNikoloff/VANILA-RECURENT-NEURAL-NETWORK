# SIMPLE-VANILA-RECURENT-NEURAL-NETWORK

Because the purpose of this project is education - I made it as simple as posible;

These are feedForward() and backPropagate() passes:

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
		err_o = (v_target - v_output) * sigmoidDerivative(v_output);                // Calculate the output layer error (step 3 for output cell).              
		err_h = dotVM(err_o, who)     * sigmoidDerivative(v_hidden);                // Calculate the hidden layer error (step 3 for hidden cell).

		who = who + deltaWeights_b(learnRate, v_hidden, err_o);                              // Update the weights for the output layer (step 4).  
		wih = wih + deltaWeights_b(learnRate, v_inputs, err_h);                              // Update the weights for the hidden layer (step 4).
	}


This is Visual Studio 2015 C++ project;
