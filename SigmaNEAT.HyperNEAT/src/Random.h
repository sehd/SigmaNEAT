#ifndef _RANDOMNESS_HEADER_H
#define _RANDOMNESS_HEADER_H

#ifdef USE_BOOST_RANDOM
    #include <boost/random.hpp>
#else
    #include <stdlib.h>
#endif

#include <vector>
#include <limits>

namespace NEAT
{

class RNG
{
    
#ifdef USE_BOOST_RANDOM
    boost::random::mt19937 gen;
#endif

public:
    // Seeds the random number generator with this value
    void Seed(long seed);

    // Seeds the random number generator with time
    void TimeSeed();

    // Returns randomly either 1 or -1
    int RandPosNeg();

    // Returns a random integer between X and Y
    int RandInt(int x, int y);

    // Returns a random number from a uniform distribution in the range of [0 .. 1]
    double RandFloat();

    // Returns a random number from a uniform distribution in the range of [-1 .. 1]
    double RandFloatSigned();

    // Returns a random number from a gaussian (normal) distribution in the range of [-1 .. 1]
    double RandGaussSigned();

    // Returns an index given a vector of probabilities
    int Roulette(std::vector<double>& a_probs);
};



} // namespace NEAT

#endif
