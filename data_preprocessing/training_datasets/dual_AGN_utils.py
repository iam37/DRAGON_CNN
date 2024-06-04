import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.modeling import models, fitting
from astropy.utils.exceptions import AstropyUserWarning
from astropy.modeling.models import Gaussian2D
from photutils.datasets import make_noise_image
import sep
import lmfit
from lmfit.lineshapes import gaussian2d, lorentzian
from lmfit import minimize, Minimizer, Parameters
import csv
from tqdm import tqdm
import time
import glob


def dualAGN_3DPSFFit(QSO, quasarName, xCenter, yCenter, contour_destination_filepath, psf_destination_filepath, radius):
    startTime = time.time()
    #print(np.shape(QSO))
    #badval = []
    #badval.append(np.where(np.isnan(QSO)))
    #QSO = np.delete(QSO, badval)
    #fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(16,10))
    #ax = ax.flatten()
    imsize = len(QSO)
    print(quasarName)
    print(contour_destination_filepath) 
    
    """Background subtraction algorithm"""
    QSO = QSO.byteswap(inplace = True).newbyteorder()
    bkgd = sep.Background(QSO)
    QSO = QSO - bkgd
    Sources_Source_Extract = sep.extract(QSO, 1.5, err = bkgd.globalrms)
    numSources = len(Sources_Source_Extract)
    print("Number of sources detected using SOURCE_EXTRACT: " + str(numSources))
    figure = plt.figure()
    plt.imshow(QSO, vmin = 0.0, vmax = 1.2, cmap = "gray_r")
    plt.xlabel("X-Axis [Pixels]")
    plt.ylabel("Y-Axis [Pixels]")
    plt.title(quasarName)
    ax = plt.gca()
    
    ax.invert_yaxis()
    plt.grid()
    
    Y, X = np.indices(QSO.shape)
    xdata = np.vstack((X.ravel(), Y.ravel()))
    p0 = [np.max(QSO), imsize/2, imsize/2, 4.52 / (2 * np.sqrt(2 * np.log(2))), 4.52 / (2 * np.sqrt(2 * np.log(2)))]
    #The following function from lmfit will also model the Gaussian data and return the Chi-Squared value of the fit
    model = lmfit.models.Gaussian2dModel(nan_policy = 'omit')
    #This to create two Gaussians so that I can fit it to the data
    #General rule: freeze some parameters, fitting, loosening parameters, and then fit again
    #Step 1: Fit 1 Gaussian
    #Fit with a single Gaussian, take that fit, let the positions vary, but start where I have the highest intensity. Freeze sigma_x and sigma_y
    #Establish a sigma_a and a sigma_b for components A and B of the two Gaussian
    #Caution: the Gaussian may fit background galaxy light. Initially make sure I throw away all image where galactic structure is visible
    #For single point-sources, the normilization A should be very close to zero.
    #Keep varying until the model is able to pick up 
    params = model.make_params(amplitude = p0[0], centerx = p0[1], sigmax = p0[3], centery =p0[2], sigmay = p0[4])
    error = np.sqrt(np.abs(QSO+1))
    result = model.fit(QSO, x=X, y=Y, params=params, weights=1/error)
    fit = model.func(X, Y, **result.best_values)
    
    
    #The following code will fit two Gaussians to the data, as described in Prof. Urry's algoritm in plot3DData()
    gauss_1 = lmfit.models.Gaussian2dModel(prefix='g1_')
    paramsOne = gauss_1.make_params(amplitude = p0[0], centerx = p0[1], sigmax = p0[3], centery =p0[2], sigmay = p0[4])
    #paramsTwo['g2_amplitude'].set(value=p0[0], min=2)
    gauss_2 = lmfit.models.Gaussian2dModel(prefix='g2_')
    paramsOne.update(gauss_2.make_params())
    paramsOne['g2_amplitude'].set(value= p0[0]) 
    #paramsOne['g1_centerx'].set(p0[1], vary = False)
    #paramsOne['g1_centery'].set(p0[2], vary = False)
    paramsOne['g2_centerx'].set(value = p0[1]+5, min = p0[1]-radius, max = p0[1] + radius)
    paramsOne['g2_centery'].set(value = p0[2], min = p0[2]-radius, max = p0[2]+radius)
    #for xCenter and yCenter parameters, find the xCenter and yCenter of the first peak, and then vary it slightly in one of the two directions to see
    #how the fit function is able to fit to the second peak. 
    #paramsOne['g2_centerx'].set(value=imsize/3, min=imsize/3.5, max=imsize/1.5)
    paramsOne['g2_sigmax'].set(value=p0[3], min = 1)
    paramsOne['g2_sigmay'].set(value=p0[4], min = 1)
    paramsOne['g2_centery'].set(value=p0[2]+5)
    #paramsTwo['g2_amplitude'].set(value=p0[0], min=2)

    twinPeakModel = gauss_1 + gauss_2
    initial_values = twinPeakModel.eval(params, x=X, y=Y)
    output_values = twinPeakModel.fit(QSO, paramsOne, x=X, y=Y)

    
    #Error calculations
    """The following script implements the Henze-Zirkler test for multivariate normality, which determines whether 
    a data set follows the null hypothesis – that the data values are normally distributed – by taking the weighted integal of
    the squared modulus of the difference between the characteristic function of squared residuals and its limit as the data
    drops off. This function will return the p-value for this test. A low p-value will mean """
    #print(multivariate_normality(QSO[int(xCenter)-radius:int(xCenter)+radius, int(yCenter)-radius:int(yCenter)+radius], alpha = 0.05)) #this 0.05 is a placeholder value that I will change based on a better understanding of
    #alpha values
    prelimArray = [] #placeholder array to hold values for later use
    for key, values in output_values.best_values.items():
        prelimArray.append(values)
    #print(prelimArray)
    g1_amplitude = output_values.params['g1_amplitude'].value
    #print("g1_amplitude: " + str(g1_amplitude))
    g1_xcenter = output_values.params['g1_centerx'].value
    #print(g1_xcenter)
    g1_ycenter = output_values.params['g1_centery'].value
    #print(g1_ycenter)
    g1_sigmaX = output_values.params['g1_sigmax'].value
    g1_sigmaY = output_values.params['g1_sigmay'].value
    print("(g1_sigmaX, g1_sigmaY)" + str((g1_sigmaX, g1_sigmaY)))
    g1_fwhmx = output_values.params['g1_fwhmx'].value
    g1_fwhmy = output_values.params['g1_fwhmy'].value
    g1_height = output_values.params['g1_height'].value
    
    g2_amplitude = output_values.params['g2_amplitude'].value
    g2_xcenter = output_values.params['g2_centerx'].value
    g2_ycenter = output_values.params['g2_centery'].value
    g2_sigmaX = output_values.params['g2_sigmax'].value
    g2_sigmaY = output_values.params['g2_sigmay'].value
    g2_fwhmx = output_values.params['g2_fwhmx'].value
    g2_fwhmy = output_values.params['g2_fwhmy'].value
    g2_height = output_values.params['g2_height'].value
    reducedChiSquare = output_values.redchi
    chiSquare = output_values.chisqr
    distance_firstPass = np.sqrt((g1_xcenter - g2_xcenter)**2 + (g1_ycenter - g2_ycenter)**2)
    sigma_r_1_firstPass = np.sqrt((g1_sigmaX)**2 + (g1_sigmaY)**2)
    sigma_r_2_firstPass = np.sqrt((g2_sigmaX)**2 + (g2_sigmaY)**2)

    doubleNucleus = False
    if distance_firstPass >= 4.0:
        doubleNucleus = True
        print ("Two Sources Detected using Gaussian Fit")
    print("Original reduced chi squared value: " + str(reducedChiSquare))
    newFitNecessary = False
    p_value = 0
    fitParams = [g1_amplitude, (g1_xcenter/g2_xcenter), (g1_ycenter/g2_ycenter), g1_xcenter, g1_ycenter, g1_sigmaX, g1_sigmaY, g2_amplitude, g2_xcenter, g2_ycenter, g2_sigmaX, g2_sigmaY,
                 reducedChiSquare, doubleNucleus, newFitNecessary, p_value, numSources]
    
    
    #based on the 25 quasar sample and the rotational simulations. The cutoff for chi-squared values should be temporarily set to 100.
    #any fit with a chi-squared above 100 and/or a sigmax/sigmay > 1.2 will be flagged by the program as potentially having a double nucleus, and will be saved for further inspection
    
    """Following section re-performs fit if reduced chi-square value is > 0.06. Fits with large reduced chi-square values
    do not accurately reflect the positions of the centers of the quasars and thus are not useful in determining the if a double nucleus quasar exists"""
    placeholder = False
    if placeholder:
        start_time = time.time()
        #Fit a single Gaussian to the data and see if it accurately finds the center of the central QSO
        results_xcenter = result.params['centerx'].value
        results_ycenter = result.params['centery'].value
        results_sigmaX = result.params['sigmax'].value
        results_sigmaY = result.params['sigmay'].value
        results_amplitude = result.params['amplitude'].value
        plt.imshow(QSO, vmin = 0.2, vmax = 0.9)
        plt.contour(X, Y, fit, cmap = "plasma")
        plt.grid()
        plt.close()
        #plt.show()
        newParams= gauss_1.make_params()
        newParams['g1_amplitude'].set(g1_amplitude, min = results_amplitude*0.2, max = results_amplitude*5, vary = True)
        newParams['g1_centerx'].set(len(X)/2, min = len(X)/2 - 2* radius, max = len(X)/2 + 2* radius, vary = True)
        newParams['g1_centery'].set(len(Y)/2, min = len(Y)/2 - 2 * radius, max = len(Y)/2 + 2 * radius, vary = True)
        #newParams['g1_sigmax'].set(results_sigmaX, min = 2.0, max = 4.0, vary = True)
        newParams['g1_sigmax'].set(2.6023513240476217, min = 2.0, max = 5.0, vary = False)
        newParams['g1_sigmay'].set(2.6023513240476217, min = 2.0, max = 5.0, vary = False)#Note: these are placeholder
        #values obtained preliminarily from the dataset. For a more rigorous fit, I will probably need to drive a better
        #feeder value for these sigma values.
        #newParams['g1_sigmay'].set(results_sigmaY, min = 2.0, max = 4.0, vary = True)
        gauss1_fit = gauss_1.fit(QSO, newParams, x=X, y=Y)
        newParams.update(gauss_2.make_params())
        newParams['g2_amplitude'].set(g1_amplitude, min = g1_amplitude*0.2, max = g1_amplitude*5, vary=True)
        newParams['g2_sigmax'].set(gauss1_fit.params['g1_sigmax'].value, min = gauss1_fit.params['g1_sigmax'].value - 0.02, max = gauss1_fit.params['g1_sigmax'].value + 0.002, vary = False)
        newParams['g2_sigmay'].set(gauss1_fit.params['g1_sigmay'].value, min = gauss1_fit.params['g1_sigmay'].value - 0.02, max = gauss1_fit.params['g1_sigmay'].value + 0.002, vary = False)
        newFits = []
        redChiFits = []
        #crazy idea to make sure that we get the right orientation of secondary fit each time. May take too much computing power:
        startTime = time.time()
        start = 0
        stop = 360
        step = 5
        i_values = np.arange(start, stop, step)
        # Calculate the corresponding values for 'newParams['g2_centerx']' and 'newParams['g2_centery']'
        g2_centerx_values = gauss1_fit.params['g1_centerx'].value + 5 * np.cos(i_values * (np.pi / 180.0))
        g2_centery_values = gauss1_fit.params['g1_centery'].value + 5 * np.sin(i_values * (np.pi / 180.0))
        # Calculate the 'min' and 'max' values for 'newParams['g2_centerx']' and 'newParams['g2_centery']'
        min_x_values = np.absolute(len(X) / 2) + 10
        max_x_values = len(X) / 2 + 2*radius * np.cos(i_values * np.pi / 180) + 1
        #print(max_x_values)
        #print(type(max_x_values))
        #print(len(max_x_values))
        min_y_values = np.absolute(len(Y) / 2) + 10
        max_y_values = len(Y) / 2 + 2*radius * np.sin(i_values * np.pi / 180) + 1
        # Set the values in 'newParams' using broadcasting
        newFits = []
        redChiFits = []
        for temp in range(len(i_values)):
            startingtime = time.time()
            sys.stdout.write('\r')
            sys.stdout.write("[{:{}}] {:.1f}%".format("="*temp, len(i_values)-1, (100/(len(i_values)-1)*temp)))
            sys.stdout.flush()
            #print("Searching for secondary fit")
            #print(max_x_values[temp])
            #print(type(max_x_values[temp]))
            newParams['g2_centerx'].set(g2_centerx_values[temp], min=min_x_values, max=max_x_values[temp], vary=True)
            newParams['g2_centery'].set(g2_centery_values[temp], min=min_y_values, max=max_y_values[temp], vary=True)
            newFit = twinPeakModel.fit(QSO, newParams, x=X, y=Y, nan_policy='omit')
            while (newFit.params['g1_centerx'].value == newFit.params['g2_centerx']) and (newFit.params['g1_centery'].value == newFit.params['g2_centery']) and (np.abs(newFit.params['g1_centerx'].value - newFit.params['g2_centerx']) <= 5) and (np.abs(newFit.params['g1_centery'].value - newFit.params['g2_centery']) <= 5):
                newParams['g2_centerx'].set(g2_centerx_values[temp] + np.random.randint(0, radius)*np.cos(temp*np.pi/180.0), min = min_x_values, max = max_x_values, vary = True)
                newParams['g2_centery'].set(g2_centery_values[temp] + np.random.randint(0, radius)*np.sin(temp*np.pi/180.0), min = min_y_values, max = max_y_values, vary = True)
                newFit = twinPeakModel.fit(QSO, newParams, x=X, y=Y, nan_policy = 'omit')
            newFits.append(newFit)
            redChiFits.append(newFit.redchi)
            #print("Revised Fit took " + str(time.time() - startingtime) + " seconds to complete.")
        redChiFits = bubbleSort(redChiFits)
        lowestRedChi = redChiFits[0]
        """
        for i in np.arange(0, 360, 20):
            startTime = time.time()
            newParams['g2_centerx'].set(gauss1_fit.params['g1_centerx'].value + 5*np.cos(i*np.pi/180), min = np.absolute(len(X)/2) , max = len(X)/2 + 25*np.cos(i*np.pi/180) + 1, vary = True)
            newParams['g2_centery'].set(gauss1_fit.params['g1_centery'].value + 5*np.sin(i*np.pi/180), min = np.absolute(len(Y)/2), max = len(Y)/2 + 25*np.sin(i*np.pi/180) + 1, vary = True)
            newFit = twinPeakModel.fit(QSO, newParams, x=X, y=Y,nan_policy = 'omit')
            redChiTemp = newFit.redchi
            newFits.append(newFit)
            redChiFits.append(redChiTemp)
            endTime = time.time()
            print("Refit process took: " + str(endTime -startTime) + " seconds")
        """
        
        #redChiFits = bubbleSort(redChiFits)
        #lowestRedChi = redChiFits[0]
        #lowestRedChi= min(redChiFits)
        #redChiFits = np.asarray(redChiFits)
        loc = np.where(redChiFits == lowestRedChi)[0][0]
        print(loc)
        print("New lowest chi squared = " + str(lowestRedChi))
        print("Secondary Fit took: " + str(time.time() - startTime) + " seconds")
        newFit = newFits[loc]
        g1_xCenter_refit = newFit.params['g1_centerx'].value
        print("g1_xCenter_refit: " + str(g1_xCenter_refit))
        g1_yCenter_refit = newFit.params['g1_centery'].value
        print("g1_xCenter_refit: " + str(g1_yCenter_refit))
        g2_xCenter_refit = newFit.params['g2_centerx'].value
        print("g2_xCenter_refit: " + str(g2_xCenter_refit))
        g2_yCenter_refit = newFit.params['g2_centery'].value
        print("g2_yCenter_refit: " + str(g2_yCenter_refit))
        g1_sigmaX_refit = newFit.params['g1_sigmax'].value
        g1_sigmaY_refit = newFit.params['g1_sigmay'].value
        g2_sigmaX_refit = newFit.params['g2_sigmax'].value
        g2_sigmaY_refit = newFit.params['g2_sigmay'].value
        distance_refit = np.sqrt((g1_xCenter_refit - g2_xCenter_refit)**2 + (g1_yCenter_refit - g2_yCenter_refit)**2)
        sigma_r_1_refit = np.sqrt((g1_sigmaX_refit)**2 + (g1_sigmaY_refit)**2)
        sigma_r_2_refit = np.sqrt((g2_sigmaX_refit)**2 + (g2_sigmaY_refit)**2)
        newFitNecessary = True

        plotFit = np.reshape(newFit.best_fit, np.shape(X))
        chiSquared2 = newFit.redchi
        #perform F-Test
        p_value = fTest(output_values.best_fit, newFit.best_fit)
        print("p_value = " + str(p_value))
        alpha = 0.05 #standard value for alpha for an f-test
        if p_value > alpha:
            print("Secondary fit disproves null hypothesis, new fit is significant")
        fitParams[15] = p_value
        fitParams[12] = chiSquared2
        if (np.round(distance_refit, 3) >= 4.0 and np.round((sigma_r_1_refit/sigma_r_2_refit), 3) >=0.7 and np.round((sigma_r_1_refit/sigma_r_2_refit), 3) <=1.4):
            doubleNucleus = True
            fitParams[13] = True
            print("Two Sources Detected with refit")
        print("--- %s seconds ---" % (time.time() - start_time))
        
    print("--- %s seconds ---" % (time.time() - startTime))
    return fitParams
def find_single_AGN(singles_filepath, singles_to_find = 1000):
    for image in glob.glob(f"{singles_filepath}*.fits"):
