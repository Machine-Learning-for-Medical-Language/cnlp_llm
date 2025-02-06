import pandas as pd

import os
import json
import random
import math
import sys
import requests
from dotenv import load_dotenv
from tqdm import tqdm

from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_recall_curve
import numpy as np
import click
import networkx as nx

from openai import OpenAI

from cnlp_llm.eval import cnlp_dataset

RANDOM = 'RANDOM'
OUT_IN = "OUTSIDEIN"
SLIDING = "SLIDING"
#RESEED = "RESEED"
SWISS = "SWISS"
GRAPH = "GRAPH"
NLTK = "nltk"

#def elo_win_probability(player_rating, opponent_rating):
#    if opponent_rating < 0 or player_rating < 0:
#        raise ValueError("ELO ratings must be non-negative")
#    return 1 / (1 + 10**(player_rating - opponent_rating))

def expected_win_probability_against_average(elo_rating):
    expected_score = 1.0 / (1.0 + (10**((1000-elo_rating)/400)))
    return expected_score
    
def elo_update_ratings(games):
    """
    # Example usage:
    games = [
        {'player1_rating': 1500, 'player2_rating': 1200, 'outcome': 1},  # Player 1 wins
        {'player1_rating': 1800, 'player2_rating': 1600, 'outcome': 0},  # Player 2 loses
        {'player1_rating': 1900, 'player2_rating': 1800, 'outcome': 2}   # Draw
    ]
    """
    for game in games:
        player1_rating = game['player1_rating']
        player2_rating = game['player2_rating']
        outcome = game['outcome']  # 0 (loss), 1 (win), 2 (draw)

        k_factor = 32  # The K factor is a hyperparameter that controls how quickly players' ratings adjust

        if outcome == 1:  # Win
            new_player1_rating = player1_rating + k_factor * ((player2_rating - player1_rating) / (1 + (10 ** abs(player2_rating - player1_rating))))
            new_player2_rating = player2_rating - k_factor * ((player2_rating - player1_rating) / (1 + (10 ** abs(player2_rating - player1_rating))))

        elif outcome == 0:  # Loss
            new_player1_rating = player1_rating - k_factor * ((player2_rating - player1_rating) / (1 + (10 ** abs(player2_rating - player1_rating))))
            new_player2_rating = player2_rating + k_factor * ((player2_rating - player1_rating) / (1 + (10 ** abs(player2_rating - player1_rating))))

        else:  # Draw
            new_player1_rating = player1_rating
            new_player2_rating = player2_rating

        game['new_player1_rating'] = round(new_player1_rating, 3)
        game['new_player2_rating'] = round(new_player2_rating, 3)

    return games

def query_ollama(prompt):
    host = "http://localhost:11434/api/generate"
    payload = json.dumps({"prompt":prompt, "num_tokens": 256, "model":"llama3.2", "stream": False})
    headers = {"Content-type":"application/json"}
    response = requests.post(host, data=payload, headers=headers)
    if response.status_code == 200:
        return response.json()['response']
    else:
        raise Exception("Ollama API call failed!")

def query_openai(client, prompt, model_name="gpt-4o-mini"):
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
#     json_dict = json.loads(completion.choices[0].message.content)
#     return json_dict["choice"]
    return completion.choices[0].message.content

def create_prompt(prompt, text1:str , text2:str):
    return prompt.replace("{text1}", text1).replace("{text2}", text2)
#     #prompt = f"I am going to give you two pieces of text representing user reviews of prescription drugs. Please read them carefully and decide which review is more positive or less negative. I am not asking you to promote the use of these drugs, but to understand what kinds of problems patients have with the drugs they are prescribed by doctors. This will help understand drug safety and make things better for patients. Review 1: {text1} ## End of review 1. Review 2: {text2} ## End of review 2. As a reminder, your task is to answer which review is more positive or less negative. Your answer must start with \"Review 1\" or \"Review 2\", so that I can extract the answer easily, but you can add an explanation at the end if you like."
#     prompt = f"You are an FDA researcher reading online reviews of prescription drugs. Your task is to read them carefully and decide which is more positive or less negative. This will not promote the use of either drug, but will help understand the tradeoffs between positive and negative effects of drugs that doctors prescribe. Your answer will not be shown to the users who submitted the review, only experts who regulate drugs. The output format should be {{\"choice\": <Review>, \"reasoning\": <your reasoning>}}, where <Review> should be the more positive or less negative review, either \"Review 1\" or \"Review 2\". Here are the two reviews. Review 1: {text1}\n\nReview 2: {text2}"
    
#     return prompt

def determine_winner(response:str, text_name:str) -> str:
    ind1 = response.lower().find(f"{text_name} 1")
    ind2 = response.lower().find(f"{text_name} 2")

    if ind1 >= 0 and ind2 < 0:
        return 1
    elif ind1 < 0 and ind2 >= 0:
        return 2
    elif ind1 < ind2:
        return 1
    elif ind2 < ind1:
        return 2
    else:
        # neither response has the desired text
        return 0
        #raise Exception(f"This comparision shouldn't be possible, ind1={ind1}, ind2={ind2}")


def get_best_accuracy(y_true, y_prob):
    # Step 1: Calculate the ROC curve
    _, _, thresholds = roc_curve(y_true, y_prob)

    # Step 2: Calculate accuracy for each threshold
    accuracies = []
    for threshold in thresholds:
        # Predicted labels based on the threshold
        y_pred = (y_prob >= threshold).astype(int)
        # Calculate accuracy
        acc = accuracy_score(y_true, y_pred)
        accuracies.append(acc)

    # Step 3: Find the threshold with the maximum accuracy
    best_index = np.argmax(accuracies)
    best_threshold = thresholds[best_index]
    best_accuracy = accuracies[best_index]

    return best_accuracy

def get_best_f1(labels, scores):
    precision, recall, thresholds = precision_recall_curve(y_true=labels, y_score=scores)
    f1s = []
    for ind,threshold in enumerate(thresholds):
        f1 = 2 * precision[ind] * recall[ind] / (precision[ind]+recall[ind])
        f1s.append(f1)

    best_index = np.argmax(f1s)
    best_f1 = f1s[best_index]
    return best_f1

class Instance:
    def __init__(self, ind: int, text: str, init: str=RANDOM):
        self.ind = ind
        self.name = f"Instance{ind}"
        self.text = text
        if init==RANDOM:
            self.elo_rating = 1000 + 100 * random.random() - 50  # random perturbance to get off perfectly even matches
        elif init==NLTK:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            sid = SentimentIntensityAnalyzer()
            ss = sid.polarity_scores(text)
            try:
                prob_pos = ss["pos"] / (ss["neg"] + ss["pos"]) # normalize by ignoring neutral probability
                self.elo_rating = 950 + 100*prob_pos
            except:
                # division by zero, which I guess means it was 100% Neutral? Random init backoff
                self.elo_rating = 1000 + 100 * random.random() - 50
        else:
            raise NotImplementedError(f"Initialization strategy {init} is not implemented yet!")
        self.wins = 0
        self.losses = 0
        self.history = []

    def update_elo(self, opponent_elo: int, win:bool):
        k_factor = 32  # The K factor is a hyperparameter that controls how quickly players' ratings adjust

        expected_score = 1.0 / (1.0 + (10**((opponent_elo-self.elo_rating)/400)))

        #update =  k_factor * ((opponent_elo - self.elo_rating) / (1 + (10 ** abs(opponent_elo - self.elo_rating))))

        if win:  # Win
            update = k_factor * (1 - expected_score)
            #self.elo_rating = self.elo_rating + update
        else:  # Loss
            update = k_factor * (-expected_score)
            #self.elo_rating = self.elo_rating + update
        
        self.elo_rating = self.elo_rating + update

class LlmTournament:
    def __init__(self, dataset, prompt, games_per_instance, query, max_players=-1, init=RANDOM):
        self.num_instances = len(dataset) if max_players==-1 else max_players
        self.prompt = prompt
        self.instances = [Instance(i, dataset[i].input, init=init) for i in range(self.num_instances)]
        self.games_per_instance = games_per_instance
        self.query = query
        self.matchup_graph = np.zeros( (self.num_instances, self.num_instances) )

    def player_ranking(self):
        return sorted(self.instances, key=lambda x: x.elo_rating)

    def play_game(self, instance1_index, instance2_index):
        instance1 = self.instances[instance1_index]
        instance2 = self.instances[instance2_index]

        competitive_prompt = create_prompt(self.prompt["prompt_template"], instance1.text, instance2.text)
        response = self.query(competitive_prompt)
        winner = 0
        attempt = 0
        max_attempts = 10
        while winner == 0:
            winner = determine_winner(response, self.prompt["text_name"])
            instance1.history.append({'opponent': instance2_index, "won": (winner==1)})
            instance2.history.append({"opponent": instance1_index, "won": (winner==2)})
            self.matchup_graph[instance1_index, instance2_index] += 1
            self.matchup_graph[instance2_index, instance1_index] += 1
            
            if winner == 1:
                return instance1_index
            elif winner == 2:
                return instance2_index
            else:
                attempt += 1
                # no winner, repeat the game?
                # print("This game had no winner")
                if attempt >= max_attempts:
                    print(f"This game has no winner in {max_attempts} attempts! Declaring it a tie.")
                    winner = -1 #instance1_index if random.random() < 0.5 else instance2_index
                    # raise Exception(f"This game has no winner in several attempts! Prompt: {competitive_prompt}, Response: {response}")

    def schedule_outside_in(self, round_num):
        next_round_matches = []
        if round_num == 0:
            for i in range(self.num_instances // 2):
                instance1_index = i
                instance2_index = self.num_instances-i-1
                next_round_matches.append((
                    (instance1_index, instance2_index),
                    self.play_game(instance1_index, instance2_index)
                ))
        else:
            instances_by_elo = self.player_ranking()
            for i in range(self.num_instances // 2):
                # this will compare highly rated vs. lowly rated players
                instance1_index = instances_by_elo[i].ind
                instance2_index = instances_by_elo[-1-i].ind
                next_round_matches.append((
                    (instance1_index, instance2_index),
                    self.play_game(instance1_index, instance2_index)
                ))

        return next_round_matches

    def schedule_randomly(self, round_num):
        """
        we use an temporary list so that we can remove players as they get scheduled and each player only plays one match per round
        """
        next_round_matches = []
        instances_by_elo = self.player_ranking()
        
        # This generates random games which isn't great for making sure everyone gets a game
        while len(instances_by_elo) > 2:
            instance1_index = instance2_index = random.randint(0, len(instances_by_elo)-1)

            # Ensure instances don't play themselves
            while instance1_index == instance2_index:
                instance2_index = random.randint(0, len(instances_by_elo)-1)

            instance1 = instances_by_elo.pop(max(instance1_index, instance2_index))
            instance2 = instances_by_elo.pop(min(instance1_index, instance2_index))
            next_round_matches.append(((instance1.ind, instance2.ind), (self.play_game(instance1.ind, instance2.ind))))
        return next_round_matches

    def schedule_sliding(self, round_num):
        """
        The sliding window strategy says, for a tournament with N ranked players, the i_th ranked player will play the i + N/2 player next.
        So the leader (0th ranked player), plays the N/2 ranked player, and so on. In an odd-sized dataset, the last place player won't get a game.
        So we take the ceiling of N/2 so instead the middle-ranked player doesn't play, and more than likely won't be in the middle twice in a row.
        As in other techniques, the first round will be random.
        """
        next_round_matches = []
        if round_num == 0:
            for i in range(self.num_instances // 2):
                instance1_index = i
                instance2_index = self.num_instances-i-1
                next_round_matches.append((
                    (instance1_index, instance2_index),
                    self.play_game(instance1_index, instance2_index)
                ))
        else:
            instances_by_elo = self.player_ranking()
            offset = math.ceil(self.num_instances/2)
            for i in range(self.num_instances // 2):
                # this will compare highly rated vs. lowly rated players
                instance1_index = instances_by_elo[i].ind
                instance2_index = instances_by_elo[i+offset].ind
                next_round_matches.append((
                    (instance1_index, instance2_index),
                    self.play_game(instance1_index, instance2_index)
                ))
        
        return next_round_matches
    
    def schedule_swiss(self, round_num):
        """
        swiss style tournament does some preliminary seeding (with random rounds?) then groups the
        competitors into groups, and does seeded brackets into 
        """
        group_size=8
        next_round_matches = []
        if round_num < 2:
            for i in range(self.num_instances // 2):
                instance1_index = i
                instance2_index = self.num_instances-i-1
                next_round_matches.append((
                    (instance1_index, instance2_index),
                    self.play_game(instance1_index, instance2_index)
                ))
        else:
            instances_by_elo = self.player_ranking()
            # figure out how many leftovers we'll have and randomly match them up
            rem = len(instances_by_elo) % group_size
            while rem >=2:
                inst1, inst2 = random.sample(instances_by_elo, 2)
                next_round_matches.append((
                    (inst1.ind, inst2.ind),
                    self.play_game(inst1.ind, inst2.ind)
                ))
                rem -=2
                instances_by_elo.remove(inst1)
                instances_by_elo.remove(inst2)

            if rem > 0:
                assert rem == 1, 'There are more remaining instances than expected after peeling off extras!'
                # one instance doesn't get a matchup
                inst = random.sample(instances_by_elo, 1)[0]
                instances_by_elo.remove(inst)

            # outer loop
            ind = 0
            while ind < len(instances_by_elo):
                for offset in range(group_size//2):
                    p1 = ind + offset
                    p2 = ind + group_size - offset - 1
                    inst1 = instances_by_elo[p1]
                    inst2 = instances_by_elo[p2]
                    next_round_matches.append((
                        (inst1.ind, inst2.ind),
                        self.play_game(inst1.ind, inst2.ind)
                    ))
                ind += group_size
            #raise NotImplementedError()

        return next_round_matches

    def schedule_graph(self, round_num):
        """
        Do 2 rounds of random match-ups, then use a distance calculation to link instances that don't have intermediate links between them
        """
        next_round_matches = []
        if round_num <= 1:
            instances_by_elo = self.player_ranking()
            with tqdm(len(instances_by_elo)) as pbar:
                while len(instances_by_elo) > 2:
                    instance1_index = instance2_index = random.randint(0, len(instances_by_elo)-1)

                    # Ensure instances don't play themselves
                    while instance1_index == instance2_index:
                        instance2_index = random.randint(0, len(instances_by_elo)-1)

                    instance1 = instances_by_elo.pop(max(instance1_index, instance2_index))
                    instance2 = instances_by_elo.pop(min(instance1_index, instance2_index))
                    next_round_matches.append(((instance1.ind, instance2.ind), (self.play_game(instance1.ind, instance2.ind))))
                    pbar.update(2)
        else:
            adj_matrix = np.array(self.matchup_graph)
            G = nx.from_numpy_array(adj_matrix)
            # Compute shortest path lengths between all pairs of nodes
            shortest_paths = dict(nx.all_pairs_shortest_path_length(G))

            # Convert shortest paths into a matrix (if needed)
            num_nodes = len(G.nodes)
            distance_matrix = np.zeros((num_nodes, num_nodes))

            for i, paths in shortest_paths.items():
                for j, distance in paths.items():
                    distance_matrix[i, j] = distance

            inds1, inds2 = np.unravel_index(distance_matrix.argsort(axis=None), distance_matrix.shape)
            inds1 = inds1.tolist()
            inds2 = inds2.tolist()
            print(f"Highest distance at this epoch is {distance_matrix[inds1[-1],inds2[-1]]}")
            # now work our way backwards through the ind matrices, assigning players to play each other as we go, and 
            # removing players from the field as necessary.
            unassigned_players = set(list(range(self.num_instances)))
            while len(inds1) > 0 and len(unassigned_players) > 1:
                ind1 = ind2 = -1
                while (ind1 not in unassigned_players or ind2 not in unassigned_players) or ind1==ind2:
                    ind1 = inds1.pop(-1)
                    ind2 = inds2.pop(-1)
                
                unassigned_players.remove(ind1)
                unassigned_players.remove(ind2)
                if distance_matrix[ind1,ind2] == 0.0:
                    ## at this point we've run out of matchups that are more than one away and we can just revert to randomness
                    break
                next_round_matches.append(((ind1, ind2), self.play_game(ind1, ind2)))
            
            if len(unassigned_players) > 1:
                # meaning we got through the list without assigning everyone (don't think this is possible??)
                #raise Exception("Didn't assign everyone!")
                while len(unassigned_players) > 1:
                    ind1,ind2 = random.sample(sorted(unassigned_players), 2)
                    unassigned_players.remove(ind1)
                    unassigned_players.remove(ind2)
                    next_round_matches.append(((ind1, ind2), self.play_game(ind1, ind2)))

                    
            elif len(unassigned_players) == 1:
                print("Must have an odd number of players")

        return next_round_matches

@click.command()
@click.argument("data_file")
@click.argument("prompt_file")
@click.option("--strategy", type=click.Choice([RANDOM, OUT_IN, SLIDING, SWISS, GRAPH], case_sensitive=False), default=RANDOM)
@click.option("--init", type=click.Choice([RANDOM, NLTK], case_sensitive=False), default=RANDOM)
@click.option("--max_instances", default=-1)
@click.option("--rounds", default=10)
@click.option("--seed", default=42)
@click.option("--task", default="sentiment")
@click.option("--model", type=click.Choice(["openai", "ollama"]))
def tournament(data_file, prompt_file, strategy=RANDOM, rounds=10, max_instances=-1, init=RANDOM, task="sentiment", seed=42, model='ollama'):

    # Print all command-line arguments
    click.echo(f"Command-line arguments: {click.get_current_context().params}")

    random.seed(seed)
    with open(prompt_file, 'rt') as f:
        prompt = json.load(f)
    dataset = cnlp_dataset(data_file, task=task)

    if model=='ollama':
        query = lambda x: query_ollama(x)
    elif model=='openai':
        load_dotenv() # get the openai API key
        client = OpenAI()
        model_name = os.getenv("OPENAI_MODEL", default="gpt-4o-mini")
        query = lambda x: query_openai(client, x, model_name)
    else:
        raise NotImplementedError(f"Model type {model} is not supported!")

    # Example usage
    tournament = LlmTournament(dataset, prompt, rounds, query, max_instances, init)
    labels = [0 if dataset[ind].target==prompt["negative_class"] else 1 for ind in range(tournament.num_instances)]
    outputs = [expected_win_probability_against_average(tournament.instances[ind].elo_rating) for ind in range(tournament.num_instances)]
    
    best_auroc = roc_auc_score(labels, outputs)
    best_acc = get_best_accuracy(labels, outputs)
    best_f1 = get_best_f1(labels, outputs)
    print(f"AUROC before starting is {best_auroc}, best accuracy is {best_acc}, best f1 is {best_f1}", flush=True)
    
    for round_num in range(tournament.games_per_instance):
        if strategy==RANDOM:
            next_round_matches = tournament.schedule_randomly(round_num)
        elif strategy==OUT_IN:
            next_round_matches = tournament.schedule_outside_in(round_num)
        elif strategy==SLIDING:
            next_round_matches = tournament.schedule_sliding(round_num)
        elif strategy==GRAPH:
            next_round_matches = tournament.schedule_graph(round_num)
        elif strategy==SWISS:
            next_round_matches = tournament.schedule_swiss(round_num)
        else:
            raise NotImplementedError(f"The strategy {strategy} is not implemented!")

        print(f"Round {round_num+1} matches:")
        for matchup, result in tqdm(next_round_matches):
            instance1 = tournament.instances[matchup[0]]
            instance2 = tournament.instances[matchup[1]]

            old_elo1 = instance1.elo_rating
            old_elo2 = instance2.elo_rating

            instance1.update_elo(opponent_elo=instance2.elo_rating, win=(result==matchup[0]))
            instance2.update_elo(opponent_elo=instance1.elo_rating, win=(result==matchup[1])) # TODO(ian) is this intentionally using the updated I1 elo?

            # print(f"Instance {matchup[0]} vs Instance {matchup[1]}: Winner is {result}")
            # print(f"  {matchup[0]} elo goes from {old_elo1} to {instance1.elo_rating}")
            # print(f"  {matchup[1]} elo goes from {old_elo2} to {instance2.elo_rating}")

        outputs = [expected_win_probability_against_average(tournament.instances[ind].elo_rating) for ind in range(tournament.num_instances)]
                   
        auroc = roc_auc_score(y_true=labels, y_score=outputs)
        acc = get_best_accuracy(labels, outputs)
        f1 = get_best_f1(labels, outputs)
        
        print(f"AUROC after round {round_num} is {auroc}, best accuracy is {acc}, best f1 is {f1}", flush=True)
        
        if auroc > best_auroc:
            best_auroc = auroc
            best_acc = acc
            best_f1 = f1

    
    print(f"Best AUROC during tournament was {best_auroc}, corresponding accuracy was {best_acc}, corresponding f1 was {best_f1}")

#     print("### Ranked elo ratings with gold labels: ###")
#     print("ind,elo,label")
#     output_list = []
#     for inst in tournament.player_ranking():
#         output = {"Instance index": inst.ind, "elo": inst.elo_rating, "label": dataset[inst.ind].target, "history": inst.history}
#         output_list.append(output)

#     print(json.dumps(output_list))

if __name__ == '__main__':
    tournament()
