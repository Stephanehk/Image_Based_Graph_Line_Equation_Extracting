# ImageBasedGraphLineEquationExtracting
This program finds the equation of a function based on an image of its graph. OpenCV is used for image preprocessing, which allows for the retrival of every point on the graph line. Polynomial regression is then used on all the points found to estimate the equation. Proprties of the graph such as minimum, maximum, and points of inflexion are also calculated. This allows for the general trend of otherwise useless graphs to be analysed. 

Awards:
1st place - 2019 Stuyvesant Hackathon


Input Image:

![Imgur](https://i.imgur.com/wQOAVBX.png)

Output:

6.333e-21 x^13  - 1.018e-17 x^12  + 6.957e-15 x^11  - 2.61e-12 x^10  + 5.562e-10 x^9

- 5.132e-08 x^8 - 5.795e-06 x^7 + 0.00269 x^6 - 0.4436 x^5 + 43.95 x^4 - 2815 x^3
              2
 + 1.147e+05 x^2 - 2.713e+06 x + 2.844e+07
 
 
Frontend:

![Imgur](https://i.imgur.com/gkoOo0W.png)
