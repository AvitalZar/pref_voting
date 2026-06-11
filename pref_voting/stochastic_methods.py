'''
    File: stochastic_methods.py
    Author: Wes Holliday (wesholliday@berkeley.edu) and Eric Pacuit (epacuit@umd.edu)
    Date: November 22, 2024
    
    Implementations of voting methods that output winners stochastically (unlike probabilistic methods, which output a probability distribution in the form of a dictionary).
'''

from pref_voting.voting_method import *
from pref_voting.iterative_methods import consensus_builder
from pref_voting.probabilistic_methods import maximal_lottery, RaDiUS
from pref_voting.grade_profiles import GradeProfile
import networkx as nx
import math
import logging
import time
import itertools
import igraph as ig
from sortedcontainers import SortedDict

@vm(name="Random Consensus Builder (Stochastic)")
def random_consensus_builder_st(profile, curr_cands=None, beta=0.5):

    """Version of the Random Consensus Builder (RCB) voting method due to Charikar et al. (https://arxiv.org/abs/2306.17838) that actually chooses a winner stochastically rather than outputting a probability distribution.

    Args:

        profile (Profile): An anonymous profile of linear orders
        curr_cands (List[int], optional): Candidates to consider. Defaults to all candidates if not provided.
        beta (float): Threshold for elimination (default 0.5). When processing candidate i, eliminates a candidate j
                    above i in the consensus building ranking if the proportion of voters preferring i to j is >= beta

    Returns:
        A sorted list of candidates.

    .. seealso::
        :meth:`pref_voting.iterative_methods.consensus_builder`
        :meth:`pref_voting.probabilistic_methods.random_consensus_builder`

    """
    consensus_building_ranking = random.choice(profile.rankings)

    return consensus_builder(profile, curr_cands=curr_cands, consensus_building_ranking=consensus_building_ranking, beta=beta)

@vm(name="Maximal Lotteries mixed with Random Consensus Builder")
def MLRCB(profile, curr_cands=None, p = 1 / math.sqrt(2), B = math.sqrt(2) - 1/2):

    """With probability p, choose the winner from the Maximal Lotteries distribution. With probability 1-p, run the stochastic version of Random Consensus Builder with beta chosen uniformly from (1/2, B). Ths method comes from Theorem 4 of Charikar et al. (https://arxiv.org/abs/2306.17838).

    Args:

        profile (Profile): An anonymous profile of linear orders
        curr_cands (List[int], optional): Candidates to consider. Defaults to all candidates if not provided.
        p (float): Probability of choosing the winner from the Maximal Lotteries distribution
        B (float): Upper bound for elimination threshold in the Random Consensus Builder method

    Returns:
        A sorted list of candidates.
    """

    if random.random() < p:
        return [maximal_lottery.choose(profile, curr_cands=curr_cands)]

    else:
        beta = random.uniform(0.5, B)
        return random_consensus_builder_st(profile, curr_cands=curr_cands, beta=beta)
    
@vm(name="Maximal Lotteries mixed with RaDiUS")
def MLRaDiUS(profile, curr_cands=None):

    """For p, B, and the probability distribution over beta given in the proof of Theorem 5 of Charikar et al. (https://arxiv.org/abs/2306.17838), choose the winner from the Maximal Lotteries distribution with probability p; with probability 1-p, run the RaDiUS method with beta chosen according to the distribution over beta.

    Args:
        profile (Profile): An anonymous profile of linear orders
        curr_cands (List[int], optional): Candidates to consider. Defaults to all candidates if not provided.

    Returns:
        A sorted list of candidates.
    """
    # Parameters
    B = 0.876353 # given in the proof of Theorem 5 of Charikar et al. (https://arxiv.org/abs/2306.17838)

    # Calculate p as per proof
    ln3 = np.log(3)
    LB = np.log((1 + B) / (1 - B))
    I = 0.5 * (LB - ln3)  # Integral value
    p = 1 / (1 + I)

    def sample_beta(B):
        # Generate a single uniform random number
        u = np.random.uniform(0, 1)

        # Compute E(u)
        Eu = np.exp(u * (LB - ln3) + ln3)

        # Compute beta sample
        beta_sample = (Eu - 1) / (Eu + 1)

        return beta_sample
    
    if random.random() < p:
        return [maximal_lottery.choose(profile, curr_cands=curr_cands)]
    
    else:
        beta = sample_beta(B)
        return [RaDiUS.choose(profile, curr_cands=curr_cands, beta=beta)]


logger = logging.getLogger("RGCR")

@vm(name="Randomized Grade Calibrated Ranking")
def RGCR(gprofile:GradeProfile, w=(lambda x: x/(1+x)), curr_cands=None):
    """
    An implementation of the cardinal ranking estimator proposed by Wang and Shah (2018) in https://arxiv.org/abs/1806.05085.
    by Avital Zar, 2026-04-21
    
    Args:
        gprofile: A profile of linear orders with associated cardinal scores (a GProfile).
        curr_cands: A list of candidates to consider. Defaults to all candidates if not provided.
        
    Returns:
        A sorted list of candidates.
    """
    
    total_start_time = time.perf_counter()
    w_results = SortedDict()

    candidates = curr_cands if curr_cands is not None else gprofile.candidates
    candidates_set = set(candidates)
    logger.info("Starting RGCR with candidates: %s", candidates)

    def _ranking_graph(gprofile:GradeProfile):
        # Helper function to create the ranking graph from the GProfile.
        rg_start_time = time.perf_counter()
        print("Ahhhhhhhh!!!!")

        num_candidates = len(candidates)
        pairwise_matrix = np.zeros((num_candidates, num_candidates), dtype=np.int8)
        cands_dict = {cand:i for i,cand in enumerate(candidates)}
        print("Boom")
        for g in gprofile._grades:
            voter = [(cands_dict[cand], score) for cand, score in g.mapping.items() if cand in candidates_set] # A generator, to prevent collapse in big input
            if not voter:
                continue
            indices = np.array([item[0] for item in voter], dtype=np.int32)
            scores = np.array([item[1] for item in voter], dtype=np.float32)

            wins_matrix = scores[:, None] > scores[None, :]
            pairwise_matrix[np.ix_(indices, indices)] |= wins_matrix.astype(np.int8)

        sources, targets = np.where(pairwise_matrix > 0)

        GB = ig.Graph(directed=True)
        GB.add_vertices(num_candidates)
        GB.vs["name"] = candidates


        batch_size = 1_000_000
        for i in range(0, len(sources), batch_size):
            batch_edges = list(zip(sources[i:i+batch_size], targets[i:i+batch_size]))
            GB.add_edges(batch_edges)
        print("Ouch")
        rg_end_time = time.perf_counter()
        logger.info("PERF: _ranking_graph execution time: %.6f seconds", rg_end_time - rg_start_time)
        return GB
    
    # This part isn't in the paper, the contrary - the paper says that ties broken is in order of the indices of the items.
    # However, such an arrangement creates a large bias in favor of the given order of candidates, which hurts the probability.
    setup_total_start = time.perf_counter()
    
    # 1. Extracting mappings
    t0 = time.perf_counter()
    gmap = [g.mapping for g in gprofile._grades]
    t_extract = time.perf_counter() - t0
    logger.info("PERF [Setup]: Extracting mappings took %.6f seconds", t_extract)
    
    # 2. Shuffling
    t0 = time.perf_counter()
    random.shuffle(gmap)
    t_shuffle = time.perf_counter() - t0
    logger.info("PERF [Setup]: Shuffling took %.6f seconds", t_shuffle)
    
    # 3. Creating GradeProfile copy
    t0 = time.perf_counter()
    Y = GradeProfile(gmap, gprofile.grades, candidates = candidates) # Create a copy of the evaluations to avoid modifying the original one.
    t_grade_profile = time.perf_counter() - t0
    logger.info("PERF [Setup]: Creating GradeProfile (Y) took %.6f seconds", t_grade_profile)

    logger.info("PERF: Initial setup TOTAL took %.6f seconds", time.perf_counter() - setup_total_start)
    
    GB = _ranking_graph(Y) # The graph g(B) which represent the ordinal ranking.
    
    graph_ops_start = time.perf_counter()
    if not GB.is_dag(): # Then someone ranked a higher-ranked item lower, in contrast to the paper's assumption.
        nx_GB = GB.to_networkx()
        nx_GB = nx.relabel_nodes(nx_GB, nx.get_node_attributes(nx_GB, 'name'))
        cycle = nx.find_cycle(nx_GB)
        nodes = [u for u, v in cycle] + [cycle[-1][1]]
        cycle_str = " -> ".join(str(node) for node in nodes)
        logger.error("Cycle detected in majority graph: %s", cycle_str)
        raise ValueError("As the algorithm assumes, there can't be cycles in voting order.")
    topo_indices = GB.topological_sorting()
    ordering = GB.vs[topo_indices]["name"]
    ordering = [c for c in ordering if c in candidates_set] # Remove candidates not in curr_cands
    logger.info("PERF: DAG check and topological sort took %.6f seconds", time.perf_counter() - graph_ops_start)
    
    logger.debug("Initial topological ordering: %s", ordering)

    def _our_can(tuple):
        # Helper random function which get two scores and return true if the first score probablistically beats the second.
        w_result = check_w(abs(tuple[0]-tuple[1]))
        prob = (1+w_result)/2 # The probability that the higher-ranked item is really better.
        result = random.random() < prob # That is, if the first one is bigger then in probability prob we return true - the first beated the second.
        if tuple[0] < tuple[1]: # If the second one is bigger, then in probability 1-prob we return true because in probability 1-prob the first beats the second.
            result = not result

        logger.debug("our_can: scores %s, prob %.4f -> flip: %g", tuple, round(prob, 4), result)
        return result

    def _find_reviewer(item):
        # Helper function which finds a random voter who graded the given item.
        reviewer = None
        for voter in Y._grades:
            if voter.has_grade(item) and voter.val(item) is not None:
                reviewer = voter
                break
        return reviewer
    
    def check_w(argument):
        # Helper function to check that w is a valid function.
        w_res = w(argument)
        if not (0 <= w_res <= 1):
            logger.error("Invalid w function: w(%g) = %g is not in [0, 1]", argument, w_res)
            raise ValueError("w must return values in [0, 1]")
        ind = w_results.bisect_left(argument)
        if ind > 0:
            k, prev = w_results.peekitem(ind-1)
            if prev > w_res:
                logger.error("Invalid w function: w is not non-decreasing. w(%g) = %g < w(%g) = %g", argument, w_res, w_results.keys()[ind-1], prev)
                raise ValueError("w must be non-decreasing")
        if ind < len(w_results):
            k, next = w_results.peekitem(ind)
            if next < w_res:
                logger.error("Invalid w function: w is not non-decreasing. w(%g) = %g > w(%g) = %g", argument, w_res, w_results.keys()[ind], next)
                raise ValueError("w must be non-decreasing")
            
        w_results[argument] = w_res
        logger.debug("Checked w(%g) = %g", argument, w_res)
        return w_res

    # Performance accumulators for the main loop
    time_in_find_reviewer = 0.0
    time_in_majority_prefers = 0.0
    time_in_list_removals = 0.0
    time_in_our_can = 0.0
    loop_start_time = time.perf_counter()

    t = 0
    while(t < len(ordering)-1):
        t_th_item = ordering[t]
        t_plus_1_th_item = ordering[t+1]

        logger.debug("Checking pair: (%s, %s) at index %g", t_th_item, t_plus_1_th_item, t)

        # Measure _find_reviewer
        t0 = time.perf_counter()
        t_reviewer = _find_reviewer(t_th_item)
        t_plus_1_reviewer = _find_reviewer(t_plus_1_th_item)
        time_in_find_reviewer += (time.perf_counter() - t0)

        # If the flipping of t and t+1 isn't a topological order, means there's no one who ranked t above t+1
        # Also if there's a reviewing for both items
        # Otherwise we continue
        
        if t_reviewer and t_plus_1_reviewer:
            # Measure majority_prefers
            t0 = time.perf_counter()
            v1 = GB.vs.find(name=t_th_item)
            v2 = GB.vs.find(name=t_plus_1_th_item)
            GB_has_edge = GB.are_adjacent(v1,v2)
            time_in_majority_prefers += (time.perf_counter() - t0)
            
            if not GB_has_edge:
                t_score = t_reviewer.val(t_th_item)
                t_plus_1_score = t_plus_1_reviewer.val(t_plus_1_th_item)

                logger.debug("Pair satisfies flip conditions. Scores: %s=%g, %s=%g", t_th_item, t_score, t_plus_1_th_item, t_plus_1_score)

                # Measure list removals (can be O(N) and slow for large lists)
                t0 = time.perf_counter()
                Y._grades.remove(t_reviewer)
                if(t_plus_1_reviewer in Y._grades): # In case we choose the same reviewer for both items.
                    Y._grades.remove(t_plus_1_reviewer)
                time_in_list_removals += (time.perf_counter() - t0)
                
                # Measure _our_can (and inner check_w)
                t0 = time.perf_counter()
                our_can_result = _our_can((t_plus_1_score, t_score))
                time_in_our_can += (time.perf_counter() - t0)
                
                if our_can_result: # If the second item ranked higher, the we flip them.
                    logger.debug("Flipping %s and %s", t_th_item, t_plus_1_th_item)
                    ordering[t], ordering[t+1] = ordering[t+1], ordering[t]
                t = t+2
            else:
                logger.debug("Skipping pair: There's a reviewer prefers %s over %s", t_th_item, t_plus_1_th_item)
                t = t+1
        else:
            logger.debug("Skipping pair: missing reviewers.")
            t = t+1
            
    loop_end_time = time.perf_counter()
    total_end_time = time.perf_counter()

    # Final Profiling Report
    logger.info("=== RGCR Performance Profiling Report ===")
    logger.info("Total execution time: %.6f seconds", total_end_time - total_start_time)
    logger.info("Main while loop total time: %.6f seconds", loop_end_time - loop_start_time)
    logger.info("  -> Time spent in _find_reviewer: %.6f seconds", time_in_find_reviewer)
    logger.info("  -> Time spent in majority_prefers: %.6f seconds", time_in_majority_prefers)
    logger.info("  -> Time spent in Y._grades.remove(): %.6f seconds", time_in_list_removals)
    logger.info("  -> Time spent in _our_can (+ check_w): %.6f seconds", time_in_our_can)
    logger.info("=========================================")

    logger.info("Final RGCR ranking: %s", ordering)
    return ordering