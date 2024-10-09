# GestureMidAir data sets

Data contain 3D hand trajectories collected with Leap Motion device. There are 13 subjects, each performs 26 interface-command gestures. Each gesture is encoded as a sequence of 3D points, representing the position of the dominant-hand forefinger.

There are 26 classes corresponding to unique gestures. See Fig. 1 of [2] for the list of gestures and their visualisation.

- Class 1: arc3Dleft
- Class 2: arc3Dright
- Class 3: caret
- Class 4: check
- Class 5: circle
- Class 6: curly-bracket-left
- Class 7: curly-bracket-right
- Class 8: delete
- Class 9: left-swipe
- Class 10: pigtail
- Class 11: poly3Dxyz
- Class 12: poly3Dxzy
- Class 13: poly3Dyxz
- Class 14: poly3Dyzx
- Class 15: poly3Dzxy
- Class 16: poly3Dzyx
- Class 17: rectangle
- Class 18: right-swipe
- Class 19: spiral
- Class 20: square-bracket-left
- Class 21: square-bracket-right
- Class 22: star
- Class 23: triangle
- Class 24: v
- Class 25: x
- Class 26: zig-zag

We make three data sets out of these data. We follow the original paper (see [2]) and use data of 8 subjects for training and 5 subjects for testing. Data of a same subjects do not appear in both train and test set.

## GestureMidAirD1

Train size: 208

Test size: 130

Missing value: No

Number of classses: 26 

Time series length: Vary

## GestureMidAirD2

Train size: 208

Test size: 130

Missing value: No

Number of classses: 26 

Time series length: Vary

## GestureMidAirD3

Train size: 208

Test size: 130

Missing value: No

Number of classses: 26 

Time series length: Vary

There is nothing to infer from the order of examples in the train and test set.

We pad NaN to the end of each time series to the length of the longest time series.

Data created by Fabio M. Caputo et al. (see [1], [2]). Data edited by Hoang Anh Dau.

[1] https://github.com/giach68/gesturesCodes

[2] Caputo, Fabio M., et al. “Comparing 3D trajectories for simple mid-air gesture recognition.” Computers & Graphics 73 (2018): 17-25.