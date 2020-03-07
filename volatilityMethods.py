import math
import numpy


def Closed_Volatility(item, N):
    # count how many rows in time series
    m = int(item['Close'].count())
    # calculate log return first
    log_return = numpy.log(item['Close']).diff()
    # calculate sigma of log return
    sigma = math.sqrt(1 / m * numpy.sum(log_return ** 2))
    # calculate close to close volatility
    return (sigma * math.sqrt(N))


# item is time series data, N is number of periods per year.
def calculate_parkinson(item, N):
    """
        Calculates the Parkinson volatility of a MultiIndexed Pandas DataFrame

        Parameters:
        item (DataFrame)

        Returns:
        float: The calculated volatility
        """

    # First we need to know how many items are in the data set
    m = int(item['High'].count())

    # We set the sum to zero and then iterate through the data
    sum = 0.0
    for index, row in item.iterrows():
        high = float(row['High'])
        low = float(row['Low'])

        if math.isnan(low):  # Check to see if there is null data in the denominator
            m = m - 1  # If so, decrease the number of items in the data set, and move to the next iteration
            continue

        sum += (math.log(high / low)) ** 2

    # We perform the rest of the calcuation and then send result value back
    return (math.sqrt(N / (1 / (4 * math.log(2))) * (1 / m) * sum))


# item is time series data, N is number of periods per year.
def calculate_garman_klass(item, N):
    """
        Calculates the Garman Klass volatility of a MultiIndexed Pandas DataFrame

        Parameters:
        item (DataFrame)

        Returns:
        float: The calculated volatility
        """

    # First we need to know how many items are in the data set
    m = int(item['High'].count())

    # We set the sum to zero and then iterate through the data
    sum = 0.0
    for index, row in item.iterrows():
        high = float(row['High'])
        low = float(row['Low'])
        closing = float(row['Close'])
        opening = float(row['Open'])

        if math.isnan(low) or math.isnan(opening):  # Check to see if there is null data in the denominator
            m = m - 1  # If so, decrease the number of items in the data set, and move to the next iteration
            continue

        sum += (0.5 * (math.log(high / low)) ** 2 - (2 * math.log(2) - 1) * (math.log(closing / opening) ** 2))

    # We perform the rest of the calcuation and then send result value back
    return math.sqrt(N / m * 1 / m * sum)
