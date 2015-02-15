from assignment1 import i_2_1, i_2_2,\
i_2_3, i_2_4, i_3_1, i_3_2, i_3_3

def continuePrompt():
    print 'Press ENTER to continue'
    raw_input()

if __name__ == '__main__':
#     first = True
#     for i in [i_2_1, i_2_2, i_2_3, i_2_4, i_3_1, i_3_2, i_3_3]:
#         if not first:
#             continuePrompt()
#         else:
#             first = False
#         print 'Running ' + i.__name__
#         i.run()
    i_2_1.run()
    continuePrompt()
    (dataset, zValues) = i_2_2.run()
    continuePrompt()
    i_2_3.run(dataset)
    continuePrompt()
    i_2_4.run(dataset, zValues)
    continuePrompt()
    i_3_1.run()
    continuePrompt()
    i_3_2.run()
    continuePrompt()
    i_3_3.run()