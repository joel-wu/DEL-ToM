import numpy as np
import random
import copy

from clause import Clause, Question
from oracle import Oracle
from dynamic_actions import *
from collections import defaultdict


def sample_question(oracle_start_state, oracle, random_actors, obj, question_idx=0):
    idx_dummy = [0]
    a1, a2, a3, a4, _ = random_actors
    questions = [Question(idx_dummy, ZeroQ(oracle, obj)),
                 Question(idx_dummy, FirstQ(oracle, a4, obj)),
                 Question(idx_dummy, SecondQ(oracle, a3, a4, obj)),
                 Question(idx_dummy, ThirdQ(oracle, a2, a3, a4, obj)),
                 Question(idx_dummy, FourthQ(oracle, a1, a2, a3, a4, obj))]
    return questions[question_idx]

#######################################
############## Chapters ###############
#######################################

def is_belief_changed(clause, is_first=False):
    action = clause.action

    # if isinstance(action, MoveAction):
    #     print(f"[CHECK] MoveAction | move={getattr(action, 'move', 'MISSING')} | {action.templates['declarative'][0]}")
    #     print(action.move)

    if isinstance(action, (ExitedAction, ExitAction, NoiseAction, EnterAction)):
        return False

    if isinstance(action, MoveAction) and not action.move:
        return False

    if isinstance(action, ObjectLocAction):
        return is_first

    return isinstance(action, (PublicTellAction, PrivateTellAction)) or \
           (isinstance(action, MoveAction) and action.move)


def write_A2_chapter(
        start_state, oracle, obj, location,
        agent_ids, all_agents,
        movements=None, exist_tell=False, questions=None,
        recorder=None                       
):
    """
    Chapter A2: 2 agents (a1, a2) move the object in the same room multiple times.
    Every time a Clause is generated, if recorder is passed, recorder() is called immediately.
    """

    # ------------------------ prepare -------------------------- 
    a1, a2 = all_agents[agent_ids[0]], all_agents[agent_ids[1]]
    outsiders = [agent for agent in all_agents if agent not in [a1, a2]]
    agent_ids = [aid + 1 for aid in agent_ids]

    # select containers: containers[0] is the initial container, 1 and 2 are used for two moves
    containers = [oracle.get_object_container(obj)]
    container_candidates = oracle.get_containers(location)[:]
    container_candidates.remove(containers[0])
    containers += random.sample(container_candidates, 2)

    # ------------------------ helper -------------------------- 
    chapter = []

    def add(clause):
        chapter.append(clause)
        if recorder:
            changed = is_belief_changed(clause, is_first=(len(chapter) == 1))
            debug_msg = f"[DEBUG] step {len(chapter)} | {clause.action.__class__.__name__} | Changed={changed}"
            recorder(changed, debug_msg)

    # ------------------------ write story -------------------------- 
    # two agents enter the room and see the object
    add(Clause(EnterAction(oracle, (a1, a2, location))))
    add(Clause(ObjectLocAction(oracle, obj, [a1, a2])))

    # a1 operation
    add(Clause(MoveAction(oracle, (a1, obj, containers[1]),
                          [a2], move=movements[0])))
    add(Clause(ExitedAction(oracle, (a1))))

    # a2 operation
    add(Clause(MoveAction(oracle, (a2, obj, containers[2]),
                          None, move=movements[1])))
    add(Clause(ExitedAction(oracle, (a2))))

    # two agents enter the waiting room
    add(Clause(EnterAction(oracle, (a1, a2, 'waiting_room'))))

    # ------------------------ tell -------------------------- 
    if exist_tell:
        tell_containers = random.sample(oracle.get_containers(location)[:], 2)
        tell_form = random.choice(range(3)) if outsiders else random.choice(range(2))

        if tell_form == 0:
            add(Clause(PublicTellAction(
                oracle, a1, obj, tell_containers[0],
                listeners=all_agents, believers=outsiders)))
            add(Clause(PrivateTellAction(
                oracle, a2, a1, obj, tell_containers[1], trust=True)))

        elif tell_form == 1:
            add(Clause(PublicTellAction(
                oracle, a2, obj, tell_containers[0],
                listeners=all_agents, believers=[a1] + outsiders)))
            add(Clause(PrivateTellAction(
                oracle, a1, a2, obj, tell_containers[1], trust=False)))

        else:  # tell_form == 2
            add(Clause(PrivateTellAction(
                oracle, a1, random.choice(outsiders),
                obj, tell_containers[0], trust=True)))

    # --------------------------------------------------
    return chapter


def write_A3_chapter(
        start_state, oracle, obj, location,
        agent_ids, all_agents,
        movements=None, exist_tell=False, questions=None,
        recorder=None):                         
    a1, a2, a3 = (all_agents[agent_ids[0]],
                  all_agents[agent_ids[1]],
                  all_agents[agent_ids[2]])
    outsiders = [ag for ag in all_agents if ag not in [a1, a2, a3]]
    agent_ids = [aid + 1 for aid in agent_ids]

    # containers
    containers = [oracle.get_object_container(obj)]
    cand = oracle.get_containers(location)[:]
    cand.remove(containers[0])
    containers += random.sample(cand, 3)

    chapter = []

    # helper
    def add(clause):
        chapter.append(clause)
        if recorder:
            changed = is_belief_changed(clause, is_first=(len(chapter) == 1))
            debug_msg = f"[DEBUG] step {len(chapter)} | {clause.action.__class__.__name__} | Changed={changed}"
            recorder(changed, debug_msg)

    # story lines
    add(Clause(EnterAction(oracle, (a1, a2, a3, location))))
    add(Clause(ObjectLocAction(oracle, obj, [a1, a2, a3])))

    # a1
    add(Clause(MoveAction(oracle, (a1, obj, containers[1]),
                          [a2, a3], move=movements[0])))
    add(Clause(ExitedAction(oracle, (a1))))
    # a2
    add(Clause(MoveAction(oracle, (a2, obj, containers[2]),
                          [a3], move=movements[1])))
    add(Clause(ExitedAction(oracle, (a2))))
    # a3
    add(Clause(MoveAction(oracle, (a3, obj, containers[3]),
                          None, move=movements[2])))
    add(Clause(ExitedAction(oracle, (a3))))

    add(Clause(EnterAction(oracle, (a1, a2, a3, 'waiting_room'))))

    # tell
    if exist_tell:
        tell_conts = random.sample(oracle.get_containers(location)[:], 2)
        tell_form = random.choice(range(4)) if outsiders else random.choice(range(2))
        if tell_form == 0:
            add(Clause(PublicTellAction(
                oracle, a2, obj, tell_conts[0],
                listeners=all_agents, believers=[a1] + outsiders)))
            add(Clause(PrivateTellAction(oracle, a3, a2,
                obj, tell_conts[1], trust=True)))
        elif tell_form == 1:
            add(Clause(PublicTellAction(
                oracle, a3, obj, tell_conts[0],
                listeners=all_agents, believers=[a1, a2] + outsiders)))
            add(Clause(PrivateTellAction(oracle, a1, a3,
                obj, tell_conts[1], trust=False)))
        elif tell_form == 2:
            add(Clause(PublicTellAction(
                oracle, a1, obj, tell_conts[0],
                listeners=all_agents, believers=outsiders)))
            add(Clause(PrivateTellAction(oracle, a3, random.choice(outsiders),
                obj, oracle.get_object_container(obj), trust=True)))
        else:  # tell_form == 3
            add(Clause(PrivateTellAction(oracle, a2, a3,
                obj, tell_conts[0], trust=False)))
            add(Clause(PrivateTellAction(oracle, a3, random.choice(outsiders),
                obj, oracle.get_object_container(obj), trust=True)))
    return chapter



def write_A4_chapter(
        start_state, oracle, obj, location,
        agent_ids, all_agents,
        movements=None, exist_tell=False, questions=None,
        recorder=None):                         
    a1, a2, a3, a4 = (all_agents[agent_ids[0]],
                      all_agents[agent_ids[1]],
                      all_agents[agent_ids[2]],
                      all_agents[agent_ids[3]])
    outsiders = [ag for ag in all_agents if ag not in [a1, a2, a3, a4]]
    agent_ids = [aid + 1 for aid in agent_ids]

    containers = [oracle.get_object_container(obj)]
    cand = oracle.get_containers(location)[:]
    cand.remove(containers[0])
    containers += random.sample(cand, 4)

    chapter = []

    def add(clause):
        chapter.append(clause)
        if recorder:
            changed = is_belief_changed(clause, is_first=(len(chapter) == 1))
            debug_msg = f"[DEBUG] step {len(chapter)} | {clause.action.__class__.__name__} | Changed={changed}"
            recorder(changed, debug_msg)

    add(Clause(EnterAction(oracle, (a1, a2, a3, a4, location))))
    add(Clause(ObjectLocAction(oracle, obj, [a1, a2, a3, a4])))

    # moves & exits
    add(Clause(MoveAction(oracle, (a1, obj, containers[1]),
                          [a2, a3, a4], move=movements[0])))
    add(Clause(ExitedAction(oracle, (a1))))
    add(Clause(MoveAction(oracle, (a2, obj, containers[2]),
                          [a3, a4], move=movements[1])))
    add(Clause(ExitedAction(oracle, (a2))))
    add(Clause(MoveAction(oracle, (a3, obj, containers[3]),
                          [a4], move=movements[2])))
    add(Clause(ExitedAction(oracle, (a3))))
    add(Clause(MoveAction(oracle, (a4, obj, containers[4]),
                          None, move=movements[3])))
    add(Clause(ExitedAction(oracle, (a4))))

    add(Clause(EnterAction(oracle, (a1, a2, a3, a4, 'waiting_room'))))

    # tell
    if exist_tell:
        tell_conts = random.sample(oracle.get_containers(location)[:], 2)
        tell_form = random.choice(range(4)) if outsiders else random.choice(range(2))
        if tell_form == 0:
            add(Clause(PublicTellAction(
                oracle, a2, obj, tell_conts[0],
                listeners=all_agents, believers=[a1] + outsiders)))
            add(Clause(PrivateTellAction(oracle, a4, a3,
                obj, tell_conts[1], trust=True)))
        elif tell_form == 1:
            add(Clause(PublicTellAction(
                oracle, a3, obj, tell_conts[0],
                listeners=all_agents, believers=[a1, a2] + outsiders)))
            add(Clause(PrivateTellAction(oracle, a1, a4,
                obj, tell_conts[1], trust=False)))
        elif tell_form == 2:
            outsider = random.choice(outsiders)
            add(Clause(PublicTellAction(
                oracle, a1, obj, tell_conts[0],
                listeners=all_agents, believers=outsiders)))
            add(Clause(PrivateTellAction(oracle, a4, outsider,
                obj, oracle.get_object_container(obj), trust=True)))
        else:  # tell_form == 3
            outsider = random.choice(outsiders)
            add(Clause(PrivateTellAction(oracle, a2, a3,
                obj, tell_conts[0], trust=False)))
            add(Clause(PrivateTellAction(oracle, a4, outsider,
                obj, oracle.get_object_container(obj), trust=True)))
    return chapter



def write_A5_chapter(
        start_state, oracle, obj, location,
        agent_ids, all_agents,
        movements=None, exist_tell=False, questions=None,
        recorder=None):                         
    a1, a2, a3, a4, a5 = (all_agents[agent_ids[0]],
                          all_agents[agent_ids[1]],
                          all_agents[agent_ids[2]],
                          all_agents[agent_ids[3]],
                          all_agents[agent_ids[4]])
    agent_ids = [aid + 1 for aid in agent_ids]

    containers = [oracle.get_object_container(obj)]
    cand = oracle.get_containers(location)[:]
    cand.remove(containers[0])
    containers += random.sample(cand, 4)

    chapter = []

    def add(clause):
        chapter.append(clause)
        if recorder:
            changed = is_belief_changed(clause, is_first=(len(chapter) == 1))
            debug_msg=f"[DEBUG] step {len(chapter)} | {clause.action.__class__.__name__} | Changed={changed}"
            recorder(changed,debug_msg)

    add(Clause(EnterAction(oracle, (a1, a2, a3, a4, a5, location))))
    add(Clause(ObjectLocAction(oracle, obj, [a1, a2, a3, a4, a5])))

    # moves & exits
    add(Clause(MoveAction(oracle, (a1, obj, containers[1]),
                          [a2, a3, a4, a5], move=movements[0])))
    add(Clause(ExitedAction(oracle, (a1))))
    add(Clause(MoveAction(oracle, (a2, obj, containers[2]),
                          [a3, a4, a5], move=movements[1])))
    add(Clause(ExitedAction(oracle, (a2))))
    add(Clause(MoveAction(oracle, (a3, obj, containers[3]),
                          [a4, a5], move=movements[2])))
    add(Clause(ExitedAction(oracle, (a3))))
    add(Clause(MoveAction(oracle, (a4, obj, containers[4]),
                          [a5], move=movements[3])))
    add(Clause(ExitedAction(oracle, (a4))))
    add(Clause(MoveAction(oracle, (a5, obj, containers[0]),
                          None, move=movements[4])))
    add(Clause(ExitedAction(oracle, (a5))))

    add(Clause(EnterAction(oracle, (a1, a2, a3, a4, a5, 'waiting_room'))))

    # tell (3 forms)
    if exist_tell:
        tell_conts = random.sample(oracle.get_containers(location)[:], 2)
        tell_form = random.choice(range(3))
        if tell_form == 0:
            add(Clause(PublicTellAction(
                oracle, a3, obj, tell_conts[0],
                listeners=all_agents, believers=[a1, a2])))
            add(Clause(PrivateTellAction(oracle, a5, a3,
                obj, tell_conts[1], trust=True)))
        elif tell_form == 1:
            add(Clause(PublicTellAction(
                oracle, a4, obj, tell_conts[0],
                listeners=all_agents, believers=[a1, a2, a3])))
            add(Clause(PrivateTellAction(oracle, a5, a1,
                obj, oracle.get_object_container(obj), trust=True)))
        else:  # tell_form == 2
            add(Clause(PrivateTellAction(oracle, a3, a1,
                obj, tell_conts[0], trust=True)))
    return chapter



#######################################
############### Tasks #################
#######################################

class Task(object):

    def __init__(self,
                 num_questions=5,
                 exit_prob=1.,
                 informant_prob=1.,
                 search_prob=1.,
                 test_cond='first order'):

        self.num_questions = num_questions

        self.search_prob = search_prob

        self.exit_inform_probs = [1 - exit_prob,
                                  exit_prob * (1 - informant_prob),
                                  exit_prob * informant_prob]
        assert sum(self.exit_inform_probs) == 1

        assert test_cond in ['first order',
                             'second order',
                             'reality',
                             'memory'], \
            "Invalid test condition: %s" % test_cond
        self.test_cond = test_cond

    def generate_story(self, world):
        raise NotImplementedError("Abstract method.")


class Specify_Tasks(Task):
    def generate_story_qs_at_end(
        self, world, tasks_per_story, tasks, num_agents=5,
        num_locations=3, statement_noise=0.1, order=0, exist_tell_in_story=False
    ):
        """
        Allows user to specify chapter and question for each task in story.

        :tasks: list with length of tasks per story. Each entry is a string in
        the set {'tb','fb','sofb'}

        :questions: list with length of tasks per story. Each entry is a string
        in the set {'memory', 'reality', 'belief', 'search'}

        :statement_noise: probability of encountering noise sentence like 'The
        dog ran through the kitchen.'
        """

        # Fetch agents and objects and select a random subset
        idx_support_dummy = [0]
        actors = world.get_actors()
        locations = world.get_locations()
        objects = world.get_objects()
        containers = world.get_containers()

        random_actors = np.random.choice(
            actors, size=num_agents, replace=False
        )
        random_locations = np.random.choice(
            locations, size=num_locations, replace=False
        )
        random_objects = np.random.choice(
            objects, size=num_locations*2, replace=False
        )
        random_containers = np.random.choice(
            containers, size=num_locations*5, replace=False
        )

        # Create the oracle
        oracle = Oracle(
            random_actors, random_locations, random_objects, random_containers
        )

        # Populate locations in the oracle with containers
        for i, random_location in enumerate(random_locations):
            location = random_location
            containers = random_containers[5*i:5*i+5]
            oracle.set_containers(location, list(containers))
            # Two of the containers have objects
            oracle.set_object_container(
                random_objects[2*i], containers[0])
            oracle.set_object_container(
                random_objects[2*i+1], containers[1])

        # Need start state for memory question
        start_state = oracle.locations.obj_containers.copy()

        # Create story by task
        chapters = {'A2': write_A2_chapter,
                    'A3': write_A3_chapter,
                    'A4': write_A4_chapter,
                    'A5': write_A5_chapter}
        story = []

        ### ===new : belief_trace container + snapshot function ===
        belief_trace = []  # store the belief snapshot for each step
        step_idx = 0
        max_belief_order=4

        def snapshot(obj_target):
            snap = {'t': step_idx,
                    'real': oracle.get_object_container(obj_target),
                    'first': {a: oracle.get_first_belief(a, obj_target) for a in random_actors},
                    'second': {f'{a1}->{a2}': oracle.get_second_belief(a1, a2, obj_target)
                               for a1 in random_actors for a2 in random_actors if a1 != a2},
                    'third': {f'{a1}->{a2}->{a3}': oracle.get_third_belief(a1, a2, a3, obj_target)
                              for a1 in random_actors for a2 in random_actors for a3 in random_actors
                              if len({a1, a2, a3}) == 3},
                    'fourth': {f'{a1}->{a2}->{a3}->{a4}': oracle.get_fourth_belief(a1, a2, a3, a4, obj_target)
                               for a1 in random_actors for a2 in random_actors
                               for a3 in random_actors for a4 in random_actors
                               if len({a1, a2, a3, a4}) == 4}
                    }
            return snap

        from collections import OrderedDict
        def recorder(belief_changed: bool, debug_msg: str):
            """
            belief_changed = True  → take a new snapshot (location / belief changed)
            belief_changed = False → copy the previous snapshot, only update the timestamp
            """
            nonlocal step_idx
            step_idx += 1

            if obj_in_question is None:
                return

            if belief_changed or not belief_trace:
                snap = snapshot(obj_in_question)
                snap = OrderedDict([
                    ("note", f"[t={step_idx}] BELIEF CHANGED"),
                    ("debug", debug_msg),
                    ("t", step_idx),
                    *snap.items()
                ])
                belief_trace.append(snap)
            else:
                dup = copy.deepcopy(belief_trace[-1])
                dup["t"] = step_idx
                dup["note"] = f"[t={step_idx}] belief UNCHANGED"
                dup["debug"] = debug_msg
                dup = OrderedDict([
                    ("note", dup["note"]),
                    ("debug", dup["debug"]),
                    ("t", dup["t"]),
                    *[(k, v) for k, v in dup.items() if k not in ("note", "debug", "t")]
                ])
                belief_trace.append(dup)

        def recorder_for_noise(index: int, debug_msg: str):
            """
            noise sentence inserted at index, corresponding belief_trace also needs to insert a
            "copy the content of index - 1, but step = index + 1" UNCHANGED trace.
            """
            if obj_in_question is None or index > len(belief_trace):
                return

            dup = copy.deepcopy(belief_trace[index - 1])
            step = index + 1
            dup["t"] = step
            dup["note"] = f"[t={step}] belief UNCHANGED"
            dup["debug"] = debug_msg
            dup = OrderedDict([
                ("note", dup["note"]),
                ("debug", dup["debug"]),
                ("t", dup["t"]),
                *[(k, v) for k, v in dup.items() if k not in ("note", "debug", "t")]
            ])
            belief_trace.insert(index, dup)

        # --------------------------------






        obj_pool = []
        obj_in_question = None

        for i in range(tasks_per_story):
            chapter = chapters[tasks[i][0]]
            location = np.random.choice(random_locations)
            obj = np.random.choice(oracle.get_objects_at_location(location))
            # Use the obj in the first chap as the target
            if i == 0:
                obj_in_question = obj

            obj_pool.append(obj)
            agent_ids = list(range(5))
            random.shuffle(agent_ids)

            # Randomly choose movements for each agent
            agent_num = int(tasks[i][0][1])
            bools = [True, False]
            movements = [random.choice(bools) for _ in range(agent_num)]
            exist_tell_in_chapter = tasks[i][1] if exist_tell_in_story else False
            # story.extend(
            #     chapter(
            #         start_state, oracle, obj, location, agent_ids, random_actors, movements=movements, exist_tell=exist_tell_in_chapter
            #     )
            # )

            # story.extend(
            #     chapter(
            #         start_state, oracle, obj, location,
            #         agent_ids, random_actors,
            #         movements=movements,
            #         exist_tell=exist_tell_in_chapter,
            #         recorder=recorder  # pass the callback
            #     )
            # )
            clause_list = chapter(
                start_state, oracle, obj, location,
                agent_ids, random_actors,
                movements=movements,
                exist_tell=exist_tell_in_chapter,
                recorder=recorder
            )

            story.extend(clause_list)

        # At the end, add noise sentences randomly
        if statement_noise:
            noisy_story = []
            prev_i = 0
            noise = [i for i
                     in range(len(story)) if np.random.rand() < statement_noise
                     ]
            for i in noise:
                # noisy_story.extend(
                #     story[prev_i:i] +
                #     [Clause(NoiseAction(random_actors,
                #             random_containers, random_objects))]
                # )
                noisy_story.extend(story[prev_i:i])

                # ---------- insert noise and take a snapshot ----------
                noise_clause = Clause(NoiseAction(
                    random_actors, random_containers, random_objects))
                noisy_story.append(noise_clause)
                debug_msg = f"[DEBUG] step {len(noisy_story)} | NoiseAction | Changed=False"
                recorder_for_noise(len(noisy_story) - 1, debug_msg)
                ###
                prev_i = i
            noisy_story.extend(story[prev_i:])

        # compute questions of all orders
        questioned_actors = copy.deepcopy(random_actors)
        random.shuffle(questioned_actors)
        for idx in range(5):
            noisy_story.append(
                sample_question(
                    start_state, oracle, questioned_actors, obj_in_question, question_idx=idx
                )
            )

        # Generate choices of containers
        choices = ', '.join(f'{chr(65+i)}. {container}' for i,
                            container in enumerate(random_containers))
        noisy_story.append('Choices: ' + choices + '\n')

        for i, trace in enumerate(belief_trace, 1):
            trace["t"] = i
            if trace["note"].endswith("BELIEF CHANGED"):
                trace["note"] = f"[t={i}] BELIEF CHANGED"
            else:
                trace["note"] = f"[t={i}] belief UNCHANGED"

            if "debug" in trace and "step" in trace["debug"]:
                parts = trace["debug"].split(" | ")
                if parts[0].startswith("[DEBUG] step "):
                    parts[0] = f"[DEBUG] step {i}"
                    trace["debug"] = " | ".join(parts)

        # find the index of the first occurrence of ObjectLocAction in belief_trace
        first_objloc_idx = None
        for i, trace in enumerate(belief_trace):
            if "ObjectLocAction" in trace.get("debug", ""):
                first_objloc_idx = i
                break

        # if found, set the previous ones to None
        if first_objloc_idx is not None:
            for i in range(first_objloc_idx):
                belief_trace[i]["real"] = None
                belief_trace[i]["first"] = None
                belief_trace[i]["second"] = None
                belief_trace[i]["third"] = None
                belief_trace[i]["fourth"] = None

        return noisy_story, belief_trace
