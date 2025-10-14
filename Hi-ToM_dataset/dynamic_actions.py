import numpy as np
import random
from itertools import combinations
from itertools import permutations


class Action(object):

    def __init__(self, templates):
        self.templates = templates

    def render_declarative(self):
        assert 'declarative' in self.templates and \
            len(self.templates['declarative']) > 0
        return np.random.choice(self.templates['declarative'])

    def render_interrogative(self):
        assert 'interrogative' in self.templates and \
            len(self.templates['interrogative']) > 0, str(self.templates)
        return np.random.choice(self.templates['interrogative'])


class ExitAction(Action):

    def __init__(self):
        templates = {
            'declarative': [
                '%s exited the %s.',
                '%s left the %s.',
                '%s went out of the %s.',
            ],
        }
        super().__init__(templates)

#########################################
############### Questions ###############
#########################################


class ZeroQ(Action):

    def __init__(self, oracle, obj):

        fill = (obj, oracle.get_object_container(obj))
        templates = {
            'interrogative': [
                'Question: Where is the %s really?\nAnswer: %s' % fill,
            ]
        }
        super().__init__(templates)


class FirstQ(Action):

    def __init__(self, oracle, agent, obj):
        fill = (agent, obj, oracle.get_first_belief(agent, obj))
        templates = {
            'interrogative': [
                'Question: Where does %s really think the %s is?\nAnswer: %s' % fill,
            ]
        }
        super().__init__(templates)


class SecondQ(Action):

    def __init__(self, oracle, a1, a2, obj):
        fill = (a1, a2, obj, oracle.get_second_belief(a1, a2, obj))
        templates = {
            'interrogative': [
                'Question: Where does %s think %s thinks the %s is?\nAnswer: %s' % fill,
            ]
        }
        super().__init__(templates)


class ThirdQ(Action):

    def __init__(self, oracle, a1, a2, a3, obj):
        fill = (a1, a2, a3, obj, oracle.get_third_belief(a1, a2, a3, obj))
        templates = {
            'interrogative': [
                'Question: Where does %s think %s thinks %s thinks the %s is?\nAnswer: %s' % fill,
            ]
        }
        super().__init__(templates)


class FourthQ(Action):

    def __init__(self, oracle, a1, a2, a3, a4, obj):
        fill = (a1, a2, a3, a4, obj,
                oracle.get_fourth_belief(a1, a2, a3, a4, obj))
        templates = {
            'interrogative': [
                'Question: Where does %s think %s thinks %s thinks %s thinks the %s is?\nAnswer: %s' % fill,
            ]
        }
        super().__init__(templates)

# class MemoryAction(Action):

#     def __init__(self, oracle_start_state, obj):
#         fill = (obj, oracle_start_state[obj])
#         templates = {
#             'interrogative': [
#                 'Where was the %s at the beginning?\t%s' % fill,
#             ]
#         }
#         super().__init__(templates)

# class LocationAction(Action):
#     def __init__(self, oracle, args):
#         """
#         Creaters string with args and modifies
#         oracle in accordance with action.
#         """
#         if len(args) == 2:
#             statement = '%s is in the %s.' % args
#             a1, loc = args
#             # may be redundant
#             oracle.set_location(a1, loc)
#         else : # 2 people
#             statement = '%s and %s are in the %s.' % args
#             a1, a2, loc = args
#             # may be redundant
#             oracle.set_location(a1, loc)
#             oracle.set_location(a2, loc)

#         templates = {
#             'declarative': [
#                 statement,
#             ]
#         }

#         super().__init__(templates)


class ObjectLocAction(Action):

    def __init__(self, oracle, obj, observers):
        container = oracle.get_object_container(obj)
        templates = {
            'declarative': [
                'The %s is in the %s.' % (obj, container),
            ]
        }

        # set first beliefs
        for observer in observers:
            oracle.set_first_belief(observer, obj, container)

        # set second beliefs
        if len(observers) >= 2:
            for observer1, observer2 in combinations(observers, 2):
                oracle.set_second_belief(observer1, observer2, obj, container)
                oracle.set_second_belief(observer2, observer1, obj, container)

        # set third beliefs
        if len(observers) >= 3:
            for chosen_observers in combinations(observers, 3):
                for observer1, observer2, observer3 in permutations(chosen_observers):
                    oracle.set_third_belief(
                        observer1, observer2, observer3, obj, container)

        # set fourth beliefs
        if len(observers) >= 4:
            for chosen_observers in combinations(observers, 4):
                for observer1, observer2, observer3, observer4 in permutations(chosen_observers):
                    oracle.set_fourth_belief(
                        observer1, observer2, observer3, observer4, obj, container)
        super().__init__(templates)


class ExitedAction(Action):

    def __init__(self, oracle, agent):
        fill = (agent, oracle.get_location(agent))

        templates = {
            'declarative': [
                '%s exited the %s.' % fill,
            ]
        }
        oracle.set_location(agent, None)
        super().__init__(templates)


class MoveAction(Action):

    def __init__(self, oracle, args, observers=None, move=True):
        agent, obj, container = args
        self.move = move
        if not move:
            location = oracle.get_container_location(container)
            templates = {
                'declarative': [
                    f'{args[0]} made no movements and stayed in the {location} for 1 minute.',
                ]
            }

        else:
            templates = {
                'declarative': [
                    '%s moved the %s to the %s.' % args,
                ]
            }

            oracle.set_object_container(obj, container)

            if not observers:
                observers = []
            observers.append(agent)

            # set first beliefs
            for observer in observers:
                oracle.set_first_belief(observer, obj, container)

            # set second beliefs
            if len(observers) >= 2:
                for observer1, observer2 in combinations(observers, 2):
                    oracle.set_second_belief(
                        observer1, observer2, obj, container)
                    oracle.set_second_belief(
                        observer2, observer1, obj, container)

            # set third beliefs
            if len(observers) >= 3:
                for chosen_observers in combinations(observers, 3):
                    for observer1, observer2, observer3 in permutations(chosen_observers):
                        oracle.set_third_belief(
                            observer1, observer2, observer3, obj, container)

            # set fourth beliefs
            if len(observers) >= 4:
                for chosen_observers in combinations(observers, 4):
                    for observer1, observer2, observer3, observer4 in permutations(chosen_observers):
                        oracle.set_fourth_belief(
                            observer1, observer2, observer3, observer4, obj, container)

        super().__init__(templates)


class PublicTellAction(Action):

    def __init__(self, oracle, speaker, obj, container, listeners=None, believers=None):
        templates = {
            'declarative': [
                '%s publicly claimed that %s is in the %s now.' % (
                    speaker, obj, container),
            ]
        }
        disbelievers = [
            listener for listener in listeners if listener not in believers]

        # All listeners would think others believe the claim
        # for believer in believers:
        #     for disbeliever in disbelievers:
        #         oracle.set_second_belief(believer, disbeliever, obj, container)
        #         oracle.set_second_belief(disbeliever, believer, obj, container)

        # A believer would think speaker also believes the obj is in container, speaker would think his words are trusted
        for believer in believers:
            oracle.set_first_belief(believer, obj, container)
            oracle.set_second_belief(believer, speaker, obj, container)
            oracle.set_second_belief(speaker, believer, obj, container)

        for disbeliever in disbelievers:
            oracle.set_second_belief(speaker, disbeliever, obj, container)

        # for listener in listeners:
        #     # the speaker believes that all the listeners believe him
        #     oracle.set_second_belief(speaker, listener, obj, container)
        #     # all listeners know the believers based on the exiting order
        #     for believer in believers:
        #         oracle.set_second_belief(listener, believer, obj, container)

        super().__init__(templates)


class PrivateTellAction(Action):

    def __init__(self, oracle, speaker, listener, obj, container, trust=True):
        templates = {
            'declarative': [
                '%s privately told %s that the %s is in the %s now.' % (
                    speaker, listener, obj, container),
            ]
        }

        # when the listener has less information (exit the room earlier), he'll trust the speaker
        if trust:
            oracle.set_first_belief(listener, obj, container)
            oracle.set_second_belief(listener, speaker, obj, container)
        oracle.set_second_belief(speaker, listener, obj, container)
        super().__init__(templates)


class EnterAction(Action):

    def __init__(self, oracle, args, observers=None, no_world_adjust=False):
        templates = {
            'declarative': [
                ', '.join(args[:-2]) + ' and ' + args[-2] +
                ' entered the ' + args[-1] + '.',
            ]
        }

        agents = args[:-1]
        location = args[-1]
        if location == 'waiting_room':
            super().__init__(templates)
            return
        for agent in agents:
            oracle.set_location(agent, location)
        objs = oracle.get_objects_at_location(location)
        observers = agents

        # agent knows location of everything
        if not no_world_adjust:
            for obj in objs:
                container = oracle.get_object_container(obj)
                # oracle.set_first_belief(agent, obj, container)
                # set first beliefs
                if len(observers) >= 1:
                    for observer in observers:
                        oracle.set_first_belief(observer, obj, container)

                # set second beliefs
                if len(observers) >= 2:
                    for observer1, observer2 in combinations(observers, 2):
                        oracle.set_second_belief(
                            observer1, observer2, obj, container)
                        oracle.set_second_belief(
                            observer2, observer1, obj, container)

                # set third beliefs
                if len(observers) >= 3:
                    for chosen_observers in combinations(observers, 3):
                        for observer1, observer2, observer3 in permutations(chosen_observers):
                            oracle.set_third_belief(
                                observer1, observer2, observer3, obj, container)

                # set fourth beliefs
                if len(observers) >= 4:
                    for chosen_observers in combinations(observers, 4):
                        for observer1, observer2, observer3, observer4 in permutations(chosen_observers):
                            oracle.set_fourth_belief(
                                observer1, observer2, observer3, observer4, obj, container)

        super().__init__(templates)


class NoiseAction(Action):

    def __init__(self, agents, containers, objects):
        animals = ['cat', 'dog', 'monkey', 'mouse']
        personal_items = ['watch', 'gloves', 'phone']
        distractors = [
            f'{random.choice(agents)} saw a {random.choice(animals)}.',
            f'{random.choice(agents)} lost his {random.choice(personal_items)}.',
            f'{random.choice(agents)} likes the {random.choice(containers)}.',
            f'{random.choice(agents)} dislikes the {random.choice(objects)}.',
        ]
        templates = {
            'declarative': [
                random.choice(distractors)
            ]
        }
        super().__init__(templates)
