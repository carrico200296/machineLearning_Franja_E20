# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 2020

@author: CARLOS
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from scipy import stats

def annr_validate(xIn, yIn, hidden_units, K, n_replicates, max_iter):
    
    loss_fn = torch.nn.MSELoss() # MSE for regression problem 
    CV = model_selection.KFold(K, shuffle=True)
    M = xIn.shape[1]
    train_error = np.empty((K,len(hidden_units)))
    test_error = np.empty((K,len(hidden_units)))
    f = 0
    #yIn = yIn.squeeze()
    
    for (k, (train_index, test_index)) in enumerate(CV.split(xIn,yIn)):
        print('\n\tCrossvalidation fold: {0}/{1}'.format(k+1,K))    
        
        X_train = torch.Tensor(xIn[train_index,:])
        y_train = torch.Tensor(yIn[train_index])
        X_test = torch.Tensor(xIn[test_index,:])
        y_test = torch.Tensor(yIn[test_index])
        
        for i in range(0,len(hidden_units)):
            # Define the model
            model = lambda: torch.nn.Sequential(
                                torch.nn.Linear(M, hidden_units[i]),
                                torch.nn.LeakyReLU(0.01),
                                #torch.nn.Tanh(),
                                torch.nn.Linear(hidden_units[i], 1),
                                )

            print('\t>> Training model with {} hidden units\n'.format(hidden_units[i]))
            
            net, final_loss, learning_curve = train_neural_net(model,
                                                               loss_fn,
                                                               X=X_train,
                                                               y=y_train,
                                                               n_replicates=n_replicates,
                                                               max_iter=max_iter)
        
            print('\n\tFinal loss with {} hidden_units = {}\n'.format(hidden_units[i],final_loss))
            
            # Determine estimated class labels for train and test set
            y_train_est = net(X_train)
            y_test_est = net(X_test)
        
            # Evaluate training and test performance
            se_train = (y_train_est.float()-y_train.float())**2 # squared error
            mse_train = (sum(se_train).type(torch.float)/len(y_train)).data.numpy() #mean
            se_test = (y_test_est.float()-y_test.float())**2 # squared error
            mse_test = (sum(se_test).type(torch.float)/len(y_test)).data.numpy() #mean
        
            train_error[f,i] = mse_train
            test_error[f,i] = mse_test
        f+=1
        
    opt_n_hidden_units = hidden_units[np.argmin(np.mean(test_error,axis=0))]
    train_err_vs_hidden_units = np.mean(train_error,axis=0)
    test_err_vs_hidden_units = np.mean(test_error,axis=0)

    return opt_n_hidden_units, train_err_vs_hidden_units, test_err_vs_hidden_units

def ann_multiclass_validate(xIn, yIn, C, hidden_units, K, n_replicates, max_iter):    

    loss_fn = torch.nn.CrossEntropyLoss()
    CV = model_selection.KFold(K, shuffle=True)
    M = xIn.shape[1]
    train_error = np.empty((K,len(hidden_units)))
    test_error = np.empty((K,len(hidden_units)))
    f = 0
    #yIn = yIn.squeeze()

    for (k, (train_index, test_index)) in enumerate(CV.split(xIn,yIn)):
        print('\n\tCrossvalidation fold: {0}/{1}'.format(k+1,K))    

        X_train = torch.Tensor(xIn[train_index,:])
        y_train = torch.Tensor(yIn[train_index,:])
        X_test = torch.Tensor(xIn[test_index,:])
        y_test = torch.Tensor(yIn[test_index,:])
        print(y_train)
        for i in range(0,len(hidden_units)):
            # Define the model
            model = lambda: torch.nn.Sequential(torch.nn.Linear(M, hidden_units[i]), 
                                                torch.nn.ReLU(), 
                                                torch.nn.Linear(hidden_units[i], C), 
                                                torch.nn.Softmax(dim=1)
                                                )
            print('\t>> Training model with {} hidden units\n'.format(hidden_units[i]))           
            net, _, _ = train_neural_net(model,
                                        loss_fn,
                                        X=X_train,
                                        y=y_train,
                                        n_replicates=n_replicates,
                                        max_iter=max_iter)
        
            print('\n\tFinal loss with {} hidden_units = {}\n'.format(hidden_units[i],final_loss))
            
            # Determine probability of each class using trained network
            softmax_logits_train = net(X_train)
            softmax_logits_test = net(X_test)
            # Get the estimated class as the class with highest probability (argmax on softmax_logits)
            y_train_est = (torch.max(softmax_logits_train, dim=1)[1]).data.numpy()
            y_test_est = (torch.max(softmax_logits_test, dim=1)[1]).data.numpy()

            # Determine errors
            e = (y_test_est != y_test)
            print('Number of miss-classifications for ANN with {} hidden units = \n\t {} out of {}'.format(hidden_units[i],sum(e),len(e)))
                
            '''
            # Evaluate training and test performance
            se_train = (y_train_est.float()-y_train.float())**2 # squared error
            mse_train = (sum(se_train).type(torch.float)/len(y_train)).data.numpy() #mean
            se_test = (y_test_est.float()-y_test.float())**2 # squared error
            mse_test = (sum(se_test).type(torch.float)/len(y_test)).data.numpy() #mean
        
            train_error[f,i] = mse_train
            test_error[f,i] = mse_test
            '''
        f+=1
    '''
    opt_n_hidden_units = hidden_units[np.argmin(np.mean(test_error,axis=0))]
    train_err_vs_hidden_units = np.mean(train_error,axis=0)
    test_err_vs_hidden_units = np.mean(test_error,axis=0)
    '''
    opt_n_hidden_units = f
    return opt_n_hidden_units


def regularized_multinominal_regression():
    '''
    # Fit multinomial logistic regression model
    regularization_strength = 1e-3
    #Try a high strength, e.g. 1e5, especially for synth2, synth3 and synth4
    mdl = lm.LogisticRegression(solver='lbfgs', multi_class='multinomial', 
                                   tol=1e-4, random_state=1, 
                                   penalty='l2', C=1/regularization_strength)
    mdl.fit(X_train,y_train)
    y_test_est = mdl.predict(X_test)
    
    test_error_rate = np.sum(y_test_est!=y_test) / len(y_test)
    
    predict = lambda x: np.argmax(mdl.predict_proba(x),1)
    plt.figure(2,figsize=(9,9))
    visualize_decision_boundary(predict, [X_train, X_test], [y_train, y_test], attributeNames, classNames)
    plt.title('LogReg decision boundaries')
    plt.show()
    '''
    return 


def train_neural_net(model, loss_fn, X, y, n_replicates=3, max_iter = 10000, tolerance=1e-6):
    """
    Args:
        model:          A function handle to make a torch.nn.Sequential.
        loss_fn:        A torch.nn-loss, e.g.  torch.nn.BCELoss() for binary 
                        binary classification, torch.nn.CrossEntropyLoss() for
                        multiclass classification, or torch.nn.MSELoss() for
                        regression (see https://pytorch.org/docs/stable/nn.html#loss-functions)
        n_replicates:   An integer specifying number of replicates to train,
                        the neural network with the lowest loss is returned.
        max_iter:       An integer specifying the maximum number of iterations
                        to do (default 10000).
        tolerenace:     A float describing the tolerance/convergence criterion
                        for minimum relative change in loss (default 1e-6)
                        
        
    Returns:
        A list of three elements:
            best_net:       A trained torch.nn.Sequential that had the lowest 
                            loss of the trained replicates
            final_loss:     An float specifying the loss of best performing net
            learning_curve: A list containing the learning curve of the best net.
    
    """
    import torch
    # Specify maximum number of iterations for training
    logging_frequency = 1000 # display the loss every 1000th iteration
    best_final_loss = 1e100
    for r in range(n_replicates):
        
        print('\tReplicate: {}/{}'.format(r+1, n_replicates))
    
        net = model()
        torch.nn.init.xavier_uniform_(net[0].weight)
        torch.nn.init.xavier_uniform_(net[2].weight)
                     
        optimizer = torch.optim.Adam(net.parameters())
        
        # Train the network while displaying and storing the loss
        print('\t\t{}\t{}\t\t{}'.format('Iter', 'Loss','Rel. loss'))
        learning_curve = [] # setup storage for loss at each step
        old_loss = 1e6
        for i in range(max_iter):
            y_est = net(X) # forward pass, predict labels on training set
            loss = loss_fn(y_est, y) # determine loss
            loss_value = loss.data.numpy() #get numpy array instead of tensor
            learning_curve.append(loss_value) # record loss for later display
            
            # Convergence check, see if the percentual loss decrease is within
            # tolerance:
            p_delta_loss = np.abs(loss_value-old_loss)/old_loss
            if p_delta_loss < tolerance: break
            old_loss = loss_value
            
            # display loss with some frequency:
            if (i != 0) & ((i+1) % logging_frequency == 0):
                print_str = '\t\t' + str(i+1) + '\t' + str(loss_value) + '\t' + str(p_delta_loss)
                print(print_str)
            # do backpropagation of loss and optimize weights 
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            
            
        # display final loss
        print('\t\tFinal loss:')
        print_str = '\t\t' + str(i+1) + '\t' + str(loss_value) + '\t' + str(p_delta_loss)
        print(print_str)
        
        if loss_value < best_final_loss: 
            best_net = net
            best_final_loss = loss_value
            best_learning_curve = learning_curve
        
    # Return the best curve along with its final loss and learing curve
    return best_net, best_final_loss, best_learning_curve


def visualize_decision_boundary(predict, 
                                 X, y, 
                                 attribute_names,
                                 class_names,
                                 train=None, test=None, 
                                 delta=5e-3,
                                 show_legend=True):
    '''
    Visualize the decision boundary of a classifier trained on a 2 dimensional
    input feature space.
    
    Creates a grid of points based on ranges of features in X, then determines
    classifier output for each point. The predictions are color-coded and plotted
    along with the data and a visualization of the partitioning in training and
    test if provided.
    
    Args:
        predict:
                A lambda function that takes the a grid of shape [M, N] as 
                input and returns the prediction of the classifier. M corre-
                sponds to the number of features (M==2 required), and N corre-
                sponding to the number of points in the grid. Can e.g. be a 
                trained PyTorch network (torch.nn.Sequential()), such as trained
                using toolbox_02450.train_neural_network, where the provided
                function would be something similar to: 
                >>> predict = lambda x: (net(torch.tensor(x, dtype=torch.float))).data.numpy()
                
        X:      A numpy array of shape (N, M), where N is the number of 
                observations and M is the number of input features (constrained
                to M==2 for this visualization).
                If X is a list of len(X)==2, then each element in X is inter-
                preted as a partition of training or test data, such that 
                X[0] is the training set and X[1] is the test set.
                
        y:      A numpy array of shape (N, 1), where N is the number of 
                observations. Each element is either 0 or 1, as the 
                visualization is constrained to a binary classification
                problem.
                If y is a list of len(y)==2, then each element in y is inter-
                preted as a partion of training or test data, such that 
                y[0] is the training set and y[1] is the test set. 
                
        attribute_names:
                A list of strings of length 2 giving the name
                of each of the M attributes in X.
                
        class_names: 
                A list of strings giving the name of each class in y.
                
        train (optional):  
                A list of indices describing the indices in X and y used for
                training the network. E.g. from the output of:
                    sklearn.model_selection.KFold(2).split(X, y)
                    
        test (optional):   
                A list of indices describing the indices in X and y used for
                testing the network (see also argument "train").
                
        delta (optional):
                A float describing the resolution of the decision
                boundary (default: 0.01). Default results grid of 100x100 that
                covers the first and second dimension range plus an additional
                25 percent.
        show_legend (optional):
                A boolean designating whether to display a legend. Defaults
                to True.
                
    Returns:
        Plots the decision boundary on a matplotlib.pyplot figure.
        
    '''
    
    import torch
    
    C = len(class_names)
    if isinstance(X, list) or isinstance(y, list):
        assert isinstance(y, list), 'If X is provided as list, y must be, too.'
        assert isinstance(y, list), 'If y is provided as list, X must be, too.'
        assert len(X)==2, 'If X is provided as a list, the length must be 2.'
        assert len(y)==2, 'If y is provided as a list, the length must be 2.'
        
        N_train, M = X[0].shape
        N_test, M = X[1].shape
        N = N_train+N_test
        grid_range = get_data_ranges(np.concatenate(X))
    else:
        N, M = X.shape
        grid_range = get_data_ranges(X)
    assert M==2, 'TwoFeatureError: Current neural_net_decision_boundary is only implemented for 2 features.'
    # Convert test/train indices to boolean index if provided:
    if train is not None or test is not None:
        assert not isinstance(X, list), 'Cannot provide indices of test and train partition, if X is provided as list of train and test partition.'
        assert not isinstance(y, list), 'Cannot provide indices of test and train partition, if y is provided as list of train and test partition.'
        assert train is not None, 'If test is provided, then train must also be provided.'
        assert test is not None, 'If train is provided, then test must also be provided.'
        train_index = np.array([(int(e) in train) for e in np.linspace(0, N-1, N)])
        test_index = np.array([(int(e) in test) for e in np.linspace(0, N-1, N)])
    
    xx = np.arange(grid_range[0], grid_range[1], delta)
    yy = np.arange(grid_range[2], grid_range[3], delta)
    # make a mesh-grid from a and b that spans the grid-range defined
    grid = np.stack(np.meshgrid(xx, yy))
    # reshape grid to be of shape "[number of feature dimensions] by [number of points in grid]"
    # this ensures that the shape fits the way the network expects input to be shaped
    # and determine estimated class label for entire featurespace by estimating
    # the label of each point in the previosly defined grid using provided
    # function predict()
    grid_predictions = predict(np.reshape(grid, (2,-1)).T)
    
    # Plot data with color designating class and transparency+shape
    # identifying partition (test/train)
    if C == 2:
        c = ['r','b']
        cmap = cm.bwr
        vmax=1
    else:
        c = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 
             'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        cmap = cm.tab10
        vmax=10
        
    s = ['o','x']; t = [.33, 1.0];
    for i in range(C):
        if train is not None and test is not None:
            for j, e in enumerate([train_index, test_index]):
                idx = (np.squeeze(y)==i) & e
                plt.plot(X[idx, 0], X[idx, 1], s[j],color=c[i], alpha=t[j])
        if isinstance(X,list) and isinstance(y, list):
            for j, (X_par, y_par) in enumerate(zip(X,y)):
                idx = np.squeeze(y_par)==i
                h = plt.plot(X_par[idx, 0], X_par[idx, 1],s[j], color=c[i], alpha=t[j])
  
    plt.xlim(grid_range[0:2])
    plt.ylim(grid_range[2:])
    plt.xlabel(attribute_names[0]);
    plt.ylabel(attribute_names[1])

    # reshape the predictions for each point in the grid to be shaped like
    # an image that corresponds to the feature-scace using the ranges that
    # defined the grid (a and b)
    decision_boundary = np.reshape(grid_predictions, (len(yy), len(xx)))
    # display the decision boundary
    ax = plt.imshow(decision_boundary, cmap=cmap, 
           extent=grid_range, vmin=0, vmax=vmax, alpha=.33, origin='lower')
    plt.axis('auto')
    if C == 2:
        plt.contour(grid[0], grid[1], decision_boundary, levels=[.5])
        plt.colorbar(ax, fraction=0.046, pad=0.04);
    if show_legend:
        plt.legend([class_names[i]+' '+e for i in range(C) for e in ['train','test']],
                   bbox_to_anchor=(1.2,1.0))
        
def draw_neural_net(weights, biases, tf, 
                    attribute_names = None,
                    figsize=(12, 12),
                    fontsizes=(15, 12)):
    '''
    Draw a neural network diagram using matplotlib based on the network weights,
    biases, and used transfer-functions. 
    
    :usage:
        >>> w = [np.array([[10, -1], [-8, 3]]), np.array([[7], [-1]])]
        >>> b = [np.array([1.5, -8]), np.array([3])]
        >>> tf = ['linear','linear']
        >>> draw_neural_net(w, b, tf)
    
    :parameters:
        - weights: list of arrays
            List of arrays, each element in list is array of weights in the 
            layer, e.g. len(weights) == 2 with a single hidden layer and
            an output layer, and weights[0].shape == (2,3) if the input 
            layer is of size two (two input features), and there are 3 hidden
            units in the hidden layer.
        - biases: list of arrays
            Similar to weights, each array in the list defines the bias
            for the given layer, such that len(biases)==2 signifies a 
            single hidden layer, and biases[0].shape==(3,) signifies that
            there are three hidden units in that hidden layer, for which
            the array defines the biases of each hidden node.
        - tf: list of strings
            List of strings defining the utilized transfer-function for each 
            layer. For use with e.g. neurolab, determine these by:
                tf = [type(e).__name__ for e in transfer_functions],
            when the transfer_functions is the parameter supplied to 
            nl.net.newff, e.g.:
                [nl.trans.TanSig(), nl.trans.PureLin()]
        - (optional) figsize: tuple of int
            Tuple of two int designating the size of the figure, 
            default is (12, 12)
        - (optional) fontsizes: tuple of int
            Tuple of two ints giving the font sizes to use for node-names and
            for weight displays, default is (15, 12).
        
    Gist originally developed by @craffel and improved by @ljhuang2017
    [https://gist.github.com/craffel/2d727968c3aaebd10359]
    
    Modifications (Nov. 7, 2018):
        * adaption for use with 02450
        * display coefficient sign and magnitude as color and 
          linewidth, respectively
        * simplifications to how the method in the gist was called
        * added optinal input of figure and font sizes
        * the usage example how  implements a recreation of the Figure 1 in
          Exercise 8 of in the DTU Course 02450
    '''

   
   
    #Determine list of layer sizes, including input and output dimensionality
    # E.g. layer_sizes == [2,2,1] has 2 inputs, 2 hidden units in a single 
    # hidden layer, and 1 outout.
    layer_sizes = [e.shape[0] for e in weights] + [weights[-1].shape[1]]
    
    # Internal renaming to fit original example of figure.
    coefs_ = weights
    intercepts_ = biases

    # Setup canvas
    fig = plt.figure(figsize=figsize)
    ax = fig.gca(); ax.axis('off');

    # the center of the leftmost node(s), rightmost node(s), bottommost node(s),
    # topmost node(s) will be placed here:
    left, right, bottom, top = [.1, .9, .1, .9]
    
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    
    # Determine normalization for width of edges between nodes:
    largest_coef = np.max([np.max(np.abs(e)) for e in weights])
    min_line_width = 1
    max_line_width = 5
    
    # Input-Arrows
    layer_top_0 = v_spacing*(layer_sizes[0] - 1)/2. + (top + bottom)/2.
    for m in range(layer_sizes[0]):
        plt.arrow(left-0.18, layer_top_0 - m*v_spacing, 0.12, 0,  
                  lw =1, head_width=0.01, head_length=0.02)

    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), 
                                v_spacing/8.,
                                color='w', ec='k', zorder=4)
            if n == 0:
                if attribute_names:
                    node_str = str(attribute_names[m])
                    
                else:
                    node_str = r'$X_{'+str(m+1)+'}$'
                plt.text(left-0.125, layer_top - m*v_spacing+v_spacing*0.1, node_str,
                         fontsize=fontsizes[0])
            elif n == n_layers -1:
                node_str =  r'$y_{'+str(m+1)+'}$'
                plt.text(n*h_spacing + left+0.10, layer_top - m*v_spacing,
                         node_str, fontsize=fontsizes[0])
                if m==layer_size-1:
                    tf_str = 'Transfer-function: \n' + tf[n-1]
                    plt.text(n*h_spacing + left, bottom, tf_str, 
                             fontsize=fontsizes[0])
            else:
                node_str = r'$H_{'+str(m+1)+','+str(n)+'}$'
                plt.text(n*h_spacing + left+0.00, 
                         layer_top - m*v_spacing+ (v_spacing/8.+0.01*v_spacing),
                         node_str, fontsize=fontsizes[0])
                if m==layer_size-1:
                    tf_str = 'Transfer-function: \n' + tf[n-1]
                    plt.text(n*h_spacing + left, bottom, 
                             tf_str, fontsize=fontsizes[0])
            ax.add_artist(circle)
            
    # Bias-Nodes
    for n, layer_size in enumerate(layer_sizes):
        if n < n_layers -1:
            x_bias = (n+0.5)*h_spacing + left
            y_bias = top + 0.005
            circle = plt.Circle((x_bias, y_bias), v_spacing/8., 
                                color='w', ec='k', zorder=4)
            plt.text(x_bias-(v_spacing/8.+0.10*v_spacing+0.01), 
                     y_bias, r'$1$', fontsize=fontsizes[0])
            ax.add_artist(circle)   
            
    # Edges
    # Edges between nodes
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                colour = 'g' if coefs_[n][m, o]>0 else 'r'
                linewidth = (coefs_[n][m, o] / largest_coef)*max_line_width+min_line_width
                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], 
                                  c=colour, linewidth=linewidth)
                ax.add_artist(line)
                xm = (n*h_spacing + left)
                xo = ((n + 1)*h_spacing + left)
                ym = (layer_top_a - m*v_spacing)
                yo = (layer_top_b - o*v_spacing)
                rot_mo_rad = np.arctan((yo-ym)/(xo-xm))
                rot_mo_deg = rot_mo_rad*180./np.pi
                xm1 = xm + (v_spacing/8.+0.05)*np.cos(rot_mo_rad)
                if n == 0:
                    if yo > ym:
                        ym1 = ym + (v_spacing/8.+0.12)*np.sin(rot_mo_rad)
                    else:
                        ym1 = ym + (v_spacing/8.+0.05)*np.sin(rot_mo_rad)
                else:
                    if yo > ym:
                        ym1 = ym + (v_spacing/8.+0.12)*np.sin(rot_mo_rad)
                    else:
                        ym1 = ym + (v_spacing/8.+0.04)*np.sin(rot_mo_rad)
                plt.text( xm1, ym1,\
                         str(round(coefs_[n][m, o],4)),\
                         rotation = rot_mo_deg, \
                         fontsize = fontsizes[1])
                
    # Edges between bias and nodes
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        if n < n_layers-1:
            layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
            layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        x_bias = (n+0.5)*h_spacing + left
        y_bias = top + 0.005 
        for o in range(layer_size_b):
            colour = 'g' if intercepts_[n][o]>0 else 'r'
            linewidth = (intercepts_[n][o] / largest_coef)*max_line_width+min_line_width
            line = plt.Line2D([x_bias, (n + 1)*h_spacing + left],
                          [y_bias, layer_top_b - o*v_spacing], 
                          c=colour,
                          linewidth=linewidth)
            ax.add_artist(line)
            xo = ((n + 1)*h_spacing + left)
            yo = (layer_top_b - o*v_spacing)
            rot_bo_rad = np.arctan((yo-y_bias)/(xo-x_bias))
            rot_bo_deg = rot_bo_rad*180./np.pi
            xo2 = xo - (v_spacing/8.+0.01)*np.cos(rot_bo_rad)
            yo2 = yo - (v_spacing/8.+0.01)*np.sin(rot_bo_rad)
            xo1 = xo2 -0.05 *np.cos(rot_bo_rad)
            yo1 = yo2 -0.05 *np.sin(rot_bo_rad)
            plt.text( xo1, yo1,\
                 str(round(intercepts_[n][o],4)),\
                 rotation = rot_bo_deg, \
                 fontsize = fontsizes[1])    
                
    # Output-Arrows
    layer_top_0 = v_spacing*(layer_sizes[-1] - 1)/2. + (top + bottom)/2.
    for m in range(layer_sizes[-1]):
        plt.arrow(right+0.015, layer_top_0 - m*v_spacing, 0.16*h_spacing, 0,  lw =1, head_width=0.01, head_length=0.02)
        
    plt.show()