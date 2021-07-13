import numpy as np
import pandas as pd

MIN_OPEN_TIME = 0 ## closePositions
MAX_OPEN_TIME = 30 ## closePositions
TAKE_PROFIT_PERCENT = 20 ## returnStopLoss
STOP_LOSS_PERCENT = 5 ## returnStopLoss
COMMISSION_LEVEL = 0.005
SCALAR_FOR_SIGMOID = 1.75
ADJUSTMENT_VALUE = 0.075
WEIGHT_REWARD = 1/0.7
WEIGHT_PENALTY = 0.7
MAX_THRESHOLD = 0.85
MIN_THRESHOLD = 0.2
LONG_THRESHOLD = 0.01
SHORT_THRESHOLD = -0.01
LONG_CONVICTION = 0.5
SHORT_CONVICTION = 0.5
indicatorsToUse = []
EPOCHS = 4

def returnLBArray(priceArray, LBperiod = 10):
    '''
    A lookback function for price to see if it has increased.
    This returns a relative price indication based on previous price.
    This is used as a Momentum indicator.
    '''
    if len(priceArray) < LBperiod:
        return(priceArray[-1])

    priceDiff = 1.1 * (priceArray[-1,:] - priceArray[-LBperiod])
    return(priceArray[-LBperiod] + priceDiff)

def returnSMAArray(priceArray, SMAperiod = 20):
    '''
    This returns an average price for a set period SMAperiod.
    This funtion is a Simple Moving Average.
    '''
    if len(priceArray) < SMAperiod:
        if len(priceArray) == 0:
            return(0)
        else:
            return(priceArray[-1])
    prices = priceArray[-SMAperiod:]
    return(prices.sum(axis=0)/SMAperiod)

def returnEMAArray(priceArray, EMAperiod = 20):
    '''
    This returns an expoentially weighted price for a set period EMAperiod.
    This funtion is an Exponential Moving Average.
    '''    

    n = priceArray.shape[0]
    if n > EMAperiod:
        priceArray = priceArray[-EMAperiod:,:]
        n = priceArray.shape[0]
    elif n != 0:
        return(priceArray[-1,:])
    else:
        return(0)

    alpha = 2 /(EMAperiod + 1.0)
    alpha_rev = 1-alpha

    pows = alpha_rev**(np.arange(n+1))
    scale_arr = 1/pows[:-1]

    offset = priceArray[-EMAperiod:,][0].reshape(-1,1)*pows[1:]
    pw0 = alpha*alpha_rev**(n-1)

    mult = (priceArray[-EMAperiod:,].transpose() * scale_arr) * pw0

    cumsums = ((priceArray[-EMAperiod:,].transpose() * scale_arr) * pw0).cumsum(axis=1)
    out = offset + cumsums*scale_arr[::-1]
    return(out[:,-1])

def returnMACDArray(priceArray, EMAPeriodShort=10, EMAPeriodLong=20):
    if len(priceArray) < EMAPeriodLong:
        if len(priceArray) == 0:
            return(0)
        else:
            return(priceArray[-1])

    p1 = returnEMAArray(priceArray, EMAPeriodShort)
    p2 = returnEMAArray(priceArray, EMAPeriodLong)

    return(priceArray[-1]*(p1/p2))

def returnSTDArray(priceArray, STDperiod = 20):
    if len(priceArray) < STDperiod:
            return(0)
    priceArray = priceArray[-STDperiod:]
    return(priceArray.std(axis=0))

def returnUpperBBArray(priceArray, BBperiod = 20, STDFactor = 1.96):    
    if len(priceArray) < BBperiod:
        if len(priceArray) == 0:
            return(0)
        else:
            return(priceArray[-1])
    prices = priceArray[-BBperiod:]
    SMA = returnSMAArray(prices, BBperiod)
    STD = returnSTDArray(prices, BBperiod)
    Upper = SMA + STD*STDFactor
    return(Upper)

def returnLowerBBArray(priceArray, BBperiod = 20, STDFactor = 1.96):
    if len(priceArray) < BBperiod:
        if len(priceArray) == 0:
            return(0)
        else:
            return(priceArray[-1])
    prices = priceArray[-BBperiod:]
    SMA = returnSMAArray(prices, BBperiod)
    STD = returnSTDArray(prices, BBperiod)
    Lower = SMA - STD*STDFactor
    return(Lower)

def returnFractalsArray(priceArray, Fperiod = 20, currentDay = 0, fractPoint = 0):

    currentDay = currentDay%Fperiod
    if len(priceArray) < Fperiod + currentDay:
        if len(priceArray) == 0:
            return(0)
        else:
            return(priceArray[-1])
    currentPrice = priceArray[-Fperiod]
    if currentDay == 0:
        previousPrice = priceArray[-Fperiod:]
    else:
        previousPrice = priceArray[-Fperiod-currentDay:-currentDay]
    High = previousPrice.min(axis=0)
    Low = previousPrice.max(axis=0)
    Close = previousPrice[-1]
    PP = (High + Low + Close)/3    
    H1 = 2*PP - Low
    S1 = 2*PP - High
    H2 = PP + 2*(PP-Low)
    S2 = PP + 2*(PP-High)
    fracs = [S2, S1, PP, H1, H2]
    return(fracs[fractPoint + 2])

def returnTMAArray(priceArray, N=9):
    """Calculate the Triangular Moving Average"""

    if len(priceArray) < N:
        if len(priceArray)==0:
            return(0)
        else:
            return priceArray[-1]

    SMA_sum = np.zeros((priceArray.shape[1],))
    for i in range(1,N):
        SMA_sum += returnSMAArray(priceArray, SMAperiod=i)
    TMA = SMA_sum / N
    return TMA

def returnPPOArray(priceArray, fastEMA = 9, slowEMA = 26, percentageThreshold = 0.1):
    """If the magnitude of the Percentage Price Oscillator is greater than the threshold,
    indicate position preference in returnPrice
    https://www.investopedia.com/articles/investing/051214/use-percentage-price-oscillator-elegant-indicator-picking-stocks.asp 
    """

    # cannot compute without at least slowEMA data points 
    if len(priceArray) < slowEMA:
        if len(priceArray) == 0:
            return (0)
        else:
            return priceArray[-1]

    # get EMA values
    fastEMAArray = returnEMAArray(priceArray, EMAperiod=fastEMA)
    slowEMAArray = returnEMAArray(priceArray, EMAperiod=slowEMA)

    # calculate ppo
    ppo = (fastEMAArray-slowEMAArray)/slowEMAArray
    currentPrice = priceArray[-1:, :]

    # boolean array of whether the magnitude of ppo is greater than the threshold
    booleanArray = np.logical_or(ppo > percentageThreshold, ppo < -percentageThreshold)
    positionPrice = currentPrice + currentPrice * ppo

    # return price with position bias if boolean array is true 
    returnPrice = np.where(booleanArray, positionPrice, currentPrice) 
    return returnPrice

def returnKArray(priceArray, lookBackPeriod):
    """Return %K of for a particular lookback period"""

    lookBackArray = priceArray[-lookBackPeriod:, :]
    currentPrice = priceArray[-1:, :]

    minPrice = np.min(lookBackArray,axis=0)
    maxPrice = np.max(lookBackArray,axis=0)

    kArray = (currentPrice - minPrice)*100/(maxPrice - minPrice)
    return kArray

def returnSOArray(priceArray, lookBackPeriod = 14, D = 3):
    """Return predicted price dependant on %K's value with respect to %D
    https://www.investopedia.com/terms/s/stochasticoscillator.asp
    """

    # not enough days too look back on for patterns
    if len(priceArray) < D+lookBackPeriod:
        if len(priceArray) == 0:
            return(0)
        else:
            return(priceArray[-1])

    # get the current value of the stochastic indicator
    currentKArray = returnKArray(priceArray, lookBackPeriod)
    # Replace 0's with the midpoint (50)
    currentKArray = np.nan_to_num(currentKArray, nan=50)

    # calculate the D-day stochastic indicator average 
    i = 99
    averageKArray = currentKArray.copy()
    for i in range(1, D):
        averageKArray += returnKArray(priceArray[-lookBackPeriod-i:-i], lookBackPeriod)
    averageKArray /= D

    # if diff is pos, current k bigger than average k -> buy
    # if diff is neg, current k smaller than average k -> sell
    diff = (currentKArray - averageKArray)/100
    currentPrice = priceArray[-1:, :]

    newPrice = currentPrice + currentPrice*diff 

    return newPrice

def returnIndicatorArray(priceArray, function, params = []):
    indicatorValue = function(priceArray, *params)
    return(indicatorValue)

indicators = {'SMA5':{'Function':returnSMAArray, 'Params':[5]}, 
                  'SMA10':{'Function':returnSMAArray, 'Params':[10]},
                  'UpperBB20STD1':{'Function':returnUpperBBArray, 'Params':[20,1]},
                  'UpperBB20STD2':{'Function':returnUpperBBArray, 'Params':[20,2]},
                  'UpperBB20STD3':{'Function':returnUpperBBArray, 'Params':[20,3]},
                  'LowerBB20STD1':{'Function':returnLowerBBArray, 'Params':[20,1]},
                  'LowerBB20STD2':{'Function':returnLowerBBArray, 'Params':[20,2]},
                  'LowerBB20STD3':{'Function':returnLowerBBArray, 'Params':[20,3]},
                  'Fractal20S2':{'Function':returnFractalsArray, 'Params':[20,0,-2]},
                  'Fractal20S1':{'Function':returnFractalsArray, 'Params':[20,0,-1]},
                  'Fractal20PP':{'Function':returnFractalsArray, 'Params':[20,0,0]},
                  'Fractal20H1':{'Function':returnFractalsArray, 'Params':[20,0,1]},
                  'Fractal20H2':{'Function':returnFractalsArray, 'Params':[20,0,2]},
                  'EMA5':{'Function':returnEMAArray, 'Params':[5]}, 
                  'EMA10':{'Function':returnEMAArray, 'Params':[10]},
                  #'MACD5_10':{'Function':returnMACDArray, 'Params':[5,10]}, 
                  #'MACD10_20':{'Function':returnMACDArray, 'Params':[10,20]}, 
                  #'MACD20_40':{'Function':returnMACDArray, 'Params':[20,40]}
                 }

tradeParameters = {'Stock':0, 'EntryTime':1, 'ExitTime':2, 'Entry':3, 'Exit':4, 
                   'Direction':5, 'Quantity':6, 'Commission':7, 'P&L':8, 'EntryThreshold': 9, 'ExitThreshold':10}

stockParameters = {'CashPosition':0, 'StockPosition':1, 'M2MPosition':2, 'P&L':3, 
                   'Commission':4, 'longThreshold':5, 'shortThreshold':6, 
                   'longConviction':7, 'shortConviction':8}

def initialiseStockParameters(numberOfStocks = 100, longThreshold = LONG_THRESHOLD, shortThreshold = SHORT_THRESHOLD, longConviction = LONG_CONVICTION, shortConviction = SHORT_CONVICTION, indicatorsToUse = []):
    '''
    This function takes a number of stocks and returns 2 numpy arrays.
    Array 1 - StockArray contains the long/shortConviction, long/shortThreshold, cashPosition, M2MPosition, stockPosition, P&L and Commission
    Array 2 - weightArray contains the weights for indicators for stocks.    
    '''

    ## a NumPy Array containing numberOfStocks x parameters
    ## Cash//StockPosition//M2MPosition//P&L//Commission//longThreshold//shortThreshold//longConviction//shortConviction
    stockArray = np.array(numberOfStocks * [[0, 0, 0, 0, 0, longThreshold, shortThreshold, longConviction, shortConviction]])

    ## a NumPy Array containing the weights for specific stocks.
    if indicatorsToUse == []:
        indicatorsToUse = list(indicators.keys())

    weightArray = np.array(numberOfStocks * [len(indicatorsToUse) * [1/len(indicatorsToUse)]])

    tradeArray = np.array([(len(indicatorsToUse) + len(tradeParameters)) * [-1]])
    return(stockArray, weightArray, tradeArray, indicatorsToUse)

def getIndicatorArray(priceArray, indicatorsToUse):

    ## Create an empty indicatorArray that is going to store the indicator values.
    indicatorArray = np.array(priceArray.shape[1] *[len(indicatorsToUse) * [0.00]])

    for indicator in range(len(indicatorsToUse)):
        indicatorToRetrieve = indicators[indicatorsToUse[indicator]]
        indicatorArray[:,indicator] = returnIndicatorArray(priceArray, indicatorToRetrieve['Function'], indicatorToRetrieve['Params'])

    return(indicatorArray)

def evaluateIndicators(priceArray, indicatorArray, weightArray):

    ## Check if the current prices are smaller than the indicatorArray prices (i.e. indicator says go up)
    evaluationArray = priceArray[-1].reshape(-1,1) < indicatorArray
    evaluationArray = evaluationArray.astype(int)
    evaluationArray[evaluationArray == 0] = -1

    ## Multiply the bool by the weight of the given Array
    evaluationArray = evaluationArray * weightArray

    ## Sum the thresholdArray
    thresholdArray = evaluationArray.sum(axis=1)

    ## Return the thresholdArray to be evaluated.
    return(thresholdArray)

def evaluateThreshold(thresholdArray, stockArray):
    longThreshold = stockArray[:, stockParameters['longThreshold']]
    shortThreshold = stockArray[:, stockParameters['shortThreshold']]

    longStocks = (thresholdArray > longThreshold).astype(int) * 1
    shortStocks = (thresholdArray < shortThreshold).astype(int) * -1

    ## Stocks where we will not trade anything - stocks cannot be both long and short by default.
    noTradeList = ((longStocks + shortStocks).astype(int)==0).astype(int)

    return(noTradeList, longStocks, shortStocks)

def adjustThreshold(stockArray, noTradeList, adjustmentFactor = ADJUSTMENT_VALUE):

    adjustment = (1 - adjustmentFactor)

    stockArray[np.where(noTradeList == 1)[0], stockParameters['longThreshold']] *= adjustment
    stockArray[np.where(noTradeList == 1)[0], stockParameters['shortThreshold']] *= adjustment

    return(stockArray)

def positionSize(positionArray, stockArray, direction, scalar = SCALAR_FOR_SIGMOID):
    '''
    This function calculates the positionSize and direction of the trade given a level of conviction.
    The tradeProportion is based on a scaled sigmoid function that considers the conviction and indicators.    
    '''
    if direction == 'Short':
        convictionParam = stockArray[:,stockParameters['shortConviction']]
    else:
        convictionParam = stockArray[:,stockParameters['longConviction']]

    positionArray = scalar * convictionParam * (np.exp(-np.logaddexp(0, -positionArray))-0.5)

    return(positionArray)

def evaluateCapacity(priceArray, stockArray, thresholdArray, longStocks, shortStocks):

    priceToday = priceArray[-1]
    OpenPosition = priceToday * stockArray[:,stockParameters['StockPosition']]
    openLong = (OpenPosition > 0).astype(int)
    openShort = (OpenPosition < 0).astype(int)
    availableCapacity = np.clip(10000 - abs(OpenPosition), 0, 10000)

    longCash = availableCapacity * positionSize(longStocks * thresholdArray, stockArray, 'Long')
    shortCash = -1 * abs(availableCapacity * positionSize(shortStocks * thresholdArray, stockArray, 'Short'))

    longCash = np.clip(longCash, 0, availableCapacity)
    shortCash = np.clip(shortCash, -availableCapacity, 0)

    sharesToLong = (longCash/priceToday).astype(int)
    sharesToShort = (shortCash/priceToday).astype(int)

    ## Check for situation where we should be long but we are currently short.
    positionToClose = ((abs(longStocks) + abs(openShort)) == 2).astype(int) + ((abs(shortStocks) + abs(openLong)) == 2).astype(int)    

    return(sharesToLong, sharesToShort, positionToClose)

def openPosition(positionArray, priceArray, stockArray, weightArray, tradeArray, indicatorArray, thresholdArray, thresholdParam = 1.05):
    if len(priceArray) == 250:
        return(stockArray, tradeArray)

    ## Determine amount of cash to spend - positionArray is directional.
    cashSpent = positionArray * priceArray[-1]

    ## Adjust your stock position in your stockArray
    stockArray[:,stockParameters['StockPosition']] += positionArray

    ## Adjust your cash position in your stockArray
    stockArray[:,stockParameters['CashPosition']] -= cashSpent

    ## Adjust your commission in your stockArray
    stockArray[:, stockParameters['Commission']] += abs(COMMISSION_LEVEL*cashSpent)

    ## Consider the weights that we need to record in our tradeArray
    weightToRecord = (positionArray != 0)
    stocksToUpdate = np.where(weightToRecord==True)[0]

    if (positionArray.sum() > 0):
        direction = 1
        stockArray[stocksToUpdate,stockParameters['longThreshold']] *= thresholdParam
        stockArray[stocksToUpdate,stockParameters['shortThreshold']] /= thresholdParam

    else:
        direction = -1
        stockArray[stocksToUpdate,stockParameters['shortThreshold']] *= thresholdParam
        stockArray[stocksToUpdate,stockParameters['longThreshold']] /= thresholdParam


    ## Record a trade into the tradeArray
    tempArray = np.zeros(shape=(weightToRecord.sum(),len(tradeParameters)+weightArray.shape[1]))
    tempArray[:,tradeParameters['EntryTime']] = len(priceArray)
    tempArray[:,tradeParameters['ExitTime']] = 0
    tempArray[:,tradeParameters['Stock']] = stocksToUpdate
    tempArray[:,tradeParameters['Entry']] = priceArray[-1][weightToRecord]
    tempArray[:,tradeParameters['Exit']] = 0
    tempArray[:,tradeParameters['Direction']] = direction
    tempArray[:,tradeParameters['Quantity']] = positionArray[weightToRecord]
    tempArray[:,tradeParameters['Commission']] = abs(COMMISSION_LEVEL*cashSpent[weightToRecord])
    tempArray[:,tradeParameters['P&L']] = 0
    tempArray[:,tradeParameters['EntryThreshold']] = thresholdArray[weightToRecord]
    tempArray[:,tradeParameters['ExitThreshold']] = 0

    indicatorDirection = ((priceArray[-1].reshape(-1,1) > indicatorArray)[weightToRecord]).astype(int)
    indicatorDirection = indicatorDirection + (indicatorDirection - 1)

    tempArray[:,len(tradeParameters):] = indicatorDirection

    tradeArray = np.vstack([tradeArray, tempArray])

    tradeArray = tradeArray[tradeArray[:,tradeParameters['Stock']]!=-1]

    return(stockArray, tradeArray)

def closePositions(priceArray, stockArray, tradeArray, thresholdArray, indicatorArray, weightArray, positionToClose, tradesStoppedOut):

    tradeArray = tradeArray.astype(float)

    ## Identify the offsetting positions by getting the index of positions to close.
    stocksToClose = np.where(positionToClose!=0)[0]
    priceToday = priceArray[-1]

    ## This variable tracks open stocks that we should close.
    stockIndex = (np.in1d(tradeArray[:,tradeParameters['Stock']], stocksToClose))
    openIndex = (tradeArray[:,tradeParameters['ExitTime']] == 0)
    openTimeIndex = (len(priceArray) - tradeArray[:,tradeParameters['EntryTime']]) > MIN_OPEN_TIME
    openTradeIndex = (stockIndex & openIndex & openTimeIndex)

    ## OpenTime and CloseTime
    closeTimeIndex = (len(priceArray) - tradeArray[:,tradeParameters['EntryTime']]) > MAX_OPEN_TIME
    openTooLong = openIndex & closeTimeIndex

    if len(priceArray) == 250:
        arrayIndex = openIndex
    else:
        ## This variable tracks which trades we need to close including those that have stopped out.
        arrayIndex = (openTradeIndex) | (openTooLong) | (tradesStoppedOut)


    ## Get prices for the ones we need to close.
    prices = priceToday[tradeArray[arrayIndex, tradeParameters['Stock']].astype(int)]

    ## Value of current position
    cash = prices * tradeArray[arrayIndex, tradeParameters['Quantity']]

    ## Record a trade into the tradeArray
    tradeArray[arrayIndex,tradeParameters['ExitTime']] = len(priceArray)
    tradeArray[arrayIndex,tradeParameters['Exit']] = prices
    tradeArray[arrayIndex,tradeParameters['Commission']] += abs(COMMISSION_LEVEL*cash)
    tradeArray[arrayIndex,tradeParameters['ExitThreshold']] = thresholdArray[tradeArray[arrayIndex, tradeParameters['Stock']].astype(int)]

    PNL = tradeArray[arrayIndex,tradeParameters['Quantity']] * (tradeArray[arrayIndex, tradeParameters['Exit']] - tradeArray[arrayIndex,tradeParameters['Entry']])
    tradeArray[arrayIndex,tradeParameters['P&L']] = PNL - tradeArray[arrayIndex,tradeParameters['Commission']]

    ## Perform an update of the stockArray thresholds.
    stockArray, weightArray = updateWeights(priceArray, stockArray, tradeArray, weightArray, indicatorArray, arrayIndex)

    return(stockArray, weightArray, tradeArray)

def updateWeights(priceArray, stockArray, tradeArray, weightArray, indicatorArray, arrayIndex, rewardParam = WEIGHT_REWARD, penaltyParam = WEIGHT_PENALTY, maxThreshold = MAX_THRESHOLD):
    '''
    Returns an updated stockArray and weightArray.
    '''

    for idx in range(len(tradeArray[arrayIndex])):
        stock = int(tradeArray[arrayIndex][idx, tradeParameters['Stock']])
        if stock != -1:
            ## We went long.
            originalParameters = tradeArray[arrayIndex][idx][len(tradeParameters):].astype(int)
            originalParameters = 2*originalParameters - 1
            newParameters = ((priceArray[-1].reshape(-1,1) > indicatorArray)[stock]).astype(int)
            newParameters = 2*newParameters - 1
            joinedParameters = originalParameters - newParameters
            doNothing = (joinedParameters == 0).astype(float)

            if tradeArray[arrayIndex][idx, tradeParameters['Quantity']] > 0:           

                ## We were profitable.
                if tradeArray[arrayIndex][idx, tradeParameters['P&L']] > 0:
                    reward = ((joinedParameters == 2)).astype(float)
                    penalty = ((joinedParameters == -2)).astype(float)
                    stockArray[stock, stockParameters['longConviction']] *= rewardParam
                else:
                    penalty = ((joinedParameters == 2)).astype(float)
                    reward = ((joinedParameters == -2)).astype(float)
                    stockArray[stock, stockParameters['longConviction']] *= penaltyParam 

                stockArray[stock, stockParameters['longThreshold']] = min(maxThreshold, stockArray[stock, stockParameters['longThreshold']]*rewardParam)

            ## We went short.
            elif tradeArray[arrayIndex][idx, tradeParameters['Quantity']] < 0:                
                ## We were profitable.
                if tradeArray[arrayIndex][idx, tradeParameters['P&L']] > 0:
                    reward = ((joinedParameters == -2)).astype(float)
                    penalty = ((joinedParameters == 2)).astype(float)
                    stockArray[stock, stockParameters['shortConviction']] *= rewardParam 
                else:
                    penalty = ((joinedParameters == 2)).astype(float)
                    reward = ((joinedParameters == -2)).astype(float)
                    stockArray[stock, stockParameters['shortConviction']] *= penaltyParam 

                stockArray[stock, stockParameters['shortThreshold']] = max(-maxThreshold, stockArray[stock, stockParameters['shortThreshold']]*rewardParam)

            stockArray[stock, stockParameters['CashPosition']] += tradeArray[arrayIndex][idx, tradeParameters['Quantity']] * tradeArray[arrayIndex][idx, tradeParameters['Exit']]
            stockArray[stock, stockParameters['StockPosition']] -= tradeArray[arrayIndex][idx, tradeParameters['Quantity']]
            stockArray[stock, stockParameters['P&L']] += tradeArray[arrayIndex][idx, tradeParameters['P&L']]
            stockArray[stock, stockParameters['Commission']] += tradeArray[arrayIndex][idx, tradeParameters['Commission']]                
            stockArray[stock, stockParameters['longConviction']] = max(1, stockArray[stock, stockParameters['longConviction']])
            stockArray[stock, stockParameters['shortConviction']] = max(1, stockArray[stock, stockParameters['shortConviction']])

            reward *= rewardParam
            reward[reward == 0] = 1
            penalty *= penaltyParam
            penalty[penalty == 0] = 1

            weightArray[stock,:] *= reward * penalty
            
            weightArray[stock,:] = np.clip(weightArray[stock,:], 0.000001, 0.5)

            weightArray[stock,:] /= weightArray[stock,:].sum()

    stockArray[:,stockParameters['longThreshold']] = np.clip(stockArray[:,stockParameters['longThreshold']], MIN_THRESHOLD, MAX_THRESHOLD)
    stockArray[:,stockParameters['shortThreshold']] = np.clip(stockArray[:,stockParameters['shortThreshold']], -MAX_THRESHOLD, -MIN_THRESHOLD)


    return(stockArray, weightArray)

def returnStopLoss(priceArray, tradeArray):
    if len(priceArray) > 0:
        priceToday = priceArray[-1]
    else:
        return(-1)

    ## Get the price for open trades.    
    ## This variable tracks whether there is an open trade.
    openIndex = (tradeArray[:,tradeParameters['ExitTime']] == 0)
    entryPrices = tradeArray[:, tradeParameters['Entry']]

    ## Percentage change of our position is indicated by the currentPosition parameter.
    currentPosition = tradeArray[:, tradeParameters['Direction']] * (priceToday[tradeArray[:,tradeParameters['Stock']].astype(int)] - entryPrices)/priceToday[tradeArray[:,tradeParameters['Stock']].astype(int)]

    ## If percentage change of our position is greater than the negative value of our stop loss, we stop out.
    tradesStoppedOut = (openIndex) & (currentPosition < -(STOP_LOSS_PERCENT/100))
    tradesInProfit = (openIndex) & (currentPosition > (TAKE_PROFIT_PERCENT/100))

    return(tradesStoppedOut & tradesInProfit)

stockArray, weightArray, tradeArray, indicatorsToUse = initialiseStockParameters()

def getMyPosition (prcSoFar):

    global stockArray
    global weightArray
    global tradeArray
    global indicatorsToUse

    if prcSoFar.shape[1] != 100:
        prcSoFar = np.array(prcSoFar).T
    
    for i in range(EPOCHS):
        ## Get the array of indicator values.
        indicatorArray = getIndicatorArray(prcSoFar, indicatorsToUse)

        ## Get the threshold array representing. 
        thresholdArray = evaluateIndicators(prcSoFar, indicatorArray, weightArray)

        noFlyList, longStocks, shortStocks = evaluateThreshold(thresholdArray, stockArray)

        stockArray = adjustThreshold(stockArray, noFlyList, adjustmentFactor = ADJUSTMENT_VALUE)

        sharesToLong, sharesToShort, positionToClose = evaluateCapacity(prcSoFar, stockArray, thresholdArray, longStocks, shortStocks)

        tradesStoppedOut = returnStopLoss(prcSoFar, tradeArray)

        stockArray, weightArray, tradeArray = closePositions(prcSoFar, stockArray, tradeArray, thresholdArray, indicatorArray, weightArray, positionToClose, tradesStoppedOut)

        stockArray, tradeArray = openPosition(sharesToShort, prcSoFar, stockArray, weightArray, tradeArray, indicatorArray, thresholdArray)

        stockArray, tradeArray = openPosition(sharesToLong, prcSoFar, stockArray, weightArray, tradeArray, indicatorArray, thresholdArray)

    return stockArray[:,stockParameters['StockPosition']].astype(int)
