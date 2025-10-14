import numpy as np


def stringify(story, exist_answer=False, order=0): # exist_answer is dummy

    lines = []

    i = 0  # The number of descriptions processed
    j = 0  # The number of lines output
    count_order = 0 

    while True:

        
        if isinstance(story[i], str):
            line = story[i]
        else:
            line = story[i].render()
            # Capitalize the line
            line = line[0].upper() + line[1:]

            # Prepend the line number
            if line.split()[0] != 'Question:' and line.split()[0] != 'Choices:':
                line = '%d %s' % (i + 1, line)
            else: # Start with 'Choice'
                if line.split()[0] == 'Choices:':
                    lines.append(line)
                    break
                else: # Start with 'Question'
                    if count_order == order:
                        lines.append(line) 
                    count_order += 1
                    i += 1
                    continue   
        lines.append(line)
        # Increment counters
        i += 1

            # Append supporting lines indices if necessary
            # if hasattr(story[i], 'idx_support') and story[i].idx_support:
            #     line += '\t%s' % ' '.join([str(x + 1)
            #                             for x in story[i].idx_support])
        
        if i >= len(story):
            break

    return lines
