import numpy as np
import torch
torch.set_printoptions(edgeitems=2, threshold=50, linewidth=75)
bikes_numpy = np.loadtxt("data/p1ch4/bike-sharing-dataset/hour-fixed.csv",
    dtype=np.float32,
    delimiter=",",
    skiprows=1,
    converters={1: lambda x: float(x[8:10])})
bikes = torch.from_numpy(bikes_numpy)
print("\n\nbikes shape and stride = {}, {}\n\n".format(bikes.shape, bikes.stride()))

# NOTE : Only difference, here would be that I'd read the initial data with pandas
#        and then still put it in pytorch and do the dimensional manipulation here
#
# '-1' : means to infer it from the other dimensions, in this case it means each day
#        for two years, hence is 730
#
# '24' : Creating new dimension of 24 for each hour
#
# 'bikes.shape[1]' : '17', how many elements are in a row, i.e. stride
#
# Net result is that

daily_bikes = bikes.view(-1, 24, bikes.shape[1])
daily_bikes.shape, daily_bikes.stride()
daily_bikes = daily_bikes.transpose(1, 2)
print("daily bikes shape and stride = {}, {}\n\n".format(daily_bikes.shape,
                                                     daily_bikes.stride()))


# b/c it is hourly data
# --> Recall that bikes is unmutate and just hourly data
first_day = bikes[:24].long()
# Weather is ordinal data
#   --> 1 = good weather
#   --> 2 =
#   --> 3 =
#   --> 4 = really bad weather
weather_onehot = torch.zeros(first_day.shape[0], 4)
# Firstday's weather
print("First day's weather : {}".format(first_day[:,9]))

# unsqeeze - makes it a column vector
#   --> b/c the weather value goes from [1->4]
#   --> Uses index to decide where to put the 'one'
weather_onehot.scatter_(
    dim=1,
    index=first_day[:,9].unsqueeze(1).long() - 1,  # adjust to being 1 indexed
    value=1.0)

# I'm not sure what this line does for me.
torch.cat((bikes[:24], weather_onehot), 1)[:1]

# Do one-hot encoding, 730 days, 4 options, 24 hours
daily_weather_onehot = torch.zeros(daily_bikes.shape[0], 4,
                                   daily_bikes.shape[2])
print("{}".format(daily_weather_onehot.shape))

# Map weather one-hot encoding for 730 days and 24 hours
daily_weather_onehot.scatter_(
    1, daily_bikes[:,9,:].long().unsqueeze(1) - 1, 1.0)
print(daily_weather_onehot.shape)

# Now append the one-hot encoding to the 17 columns already present
daily_bikes = torch.cat((daily_bikes, daily_weather_onehot), dim=1)

# An alternative way treating the weather situation, instead of one-hot encoding
# make the min value 0 and then normalize
daily_bikes[:, 9, :] = (daily_bikes[:, 9, :] - 1.0) / 3.0


temp = daily_bikes[:, 10, :]
temp_min = torch.min(temp)
temp_max = torch.max(temp)

daily_bikes[:, 10, :] = ((daily_bikes[:, 10, :] - temp_min)
                         / (temp_max - temp_min))
temp = daily_bikes[:, 10, :]
daily_bikes[:, 10, :] = ((daily_bikes[:, 10, :] - torch.mean(temp))
                         / torch.std(temp))

