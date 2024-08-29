import clr
import sys
import os

# Adding paths to the required .NET assemblies
user_directory = os.path.expanduser("~")

sys.path.append(user_directory + '\\.nuget\\packages\microsoft.ml.probabilistic\\0.4.2403.801\\lib\\netstandard2.0')
clr.AddReference("Microsoft.ML.Probabilistic")

sys.path.append(user_directory + '\\.nuget\\packages\microsoft.ml.probabilistic.compiler\\0.4.2403.801\\lib\\netstandard2.0')
clr.AddReference("Microsoft.ML.Probabilistic.Compiler")

sys.path.append(user_directory + '\\.nuget\\packages\\system.codedom\\8.0.0\\lib\\net462')
clr.AddReference("System.CodeDom")

# Importing necessary classes from the .NET assemblies
from Microsoft.ML.Probabilistic import *
from Microsoft.ML.Probabilistic.Distributions import VectorGaussian
from Microsoft.ML.Probabilistic.Models import Variable, InferenceEngine


def rankRound(round, type, beta = 8.3333333333333):
    """
    Uses the outcome of a round to update speaker ratings

    Parameters:
    round (list): A list of four teams of speakers, speakers are represented by a tuple of their mean and variance
    type (str): If the round is an inround, outround, or final
    beta (float): A constant for adjusting the "noisiness" of the round (i.e. poor judging, poor motion balance)
    
    Returns:
    list: A list of updated ratings for each speaker after the round represented by a tuple of their mean and variance
    """

    # Outrounds tend to have higher quality judges and motions, thus are less noisy 
    if type == 'outround':
        beta = beta / 3
    elif type == 'final':
        beta = beta / 4

    # Individual speakers's skills are represented as a Gaussian
    # Create an array of skill objects for each speaker
    team_speakers = [
        Variable.GaussianFromMeanAndVariance(round[i][j][0], round[i][j][1])
        for i in range(4) for j in range(2)
    ]

    # The performance of teams is represented as a guassian that is a noisy convolution of the Gaussians the the individual speakers
    # Using speakers from the array create an array of teams
    teams = [
        Variable.GaussianFromMeanAndVariance(
            team_speakers[2*i].op_Addition(team_speakers[2*i], team_speakers[2*i+1]), beta
        )
        for i in range(4)
    ]
    
    # To "win against" is have a performance greater than the other teams
    # The number of "win against" comparisons depends on the type of round 
    Variable.ConstrainTrue(teams[0].op_GreaterThan(teams[0],teams[2]))
    Variable.ConstrainTrue(teams[0].op_GreaterThan(teams[0],teams[3]))
    
    if type == 'inround':
        Variable.ConstrainTrue(teams[0].op_GreaterThan(teams[0],teams[1]))
        Variable.ConstrainTrue(teams[1].op_GreaterThan(teams[1],teams[2]))
        Variable.ConstrainTrue(teams[1].op_GreaterThan(teams[1],teams[3]))
        Variable.ConstrainTrue(teams[2].op_GreaterThan(teams[2],teams[3]))
        
    if type == 'outround':
        Variable.ConstrainTrue(teams[1].op_GreaterThan(teams[1],teams[2]))
        Variable.ConstrainTrue(teams[1].op_GreaterThan(teams[1],teams[3]))

    if type == 'final':
        Variable.ConstrainTrue(teams[0].op_GreaterThan(teams[0],teams[1]))

    # Run inference engine
    engine = InferenceEngine()

    # Update ratings for each speaker
    round_update = [[], [], [], []]

    # 
    for rank in range(4):
        for speaker in range(2):
            speaker_info = engine.Infer(round[rank][speaker])
            speaker_rating = speaker_info.GetMean()
            speaker_uncertainty = speaker_info.GetVariance()
            round_update[rank].append((speaker_rating, speaker_uncertainty))

    return round_update