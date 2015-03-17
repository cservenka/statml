import iii_1_1, iii_2_1, iii_2_2, iii_2_3_1

def continuePrompt():
    print 'Press ENTER to continue'
    raw_input()

if __name__ == '__main__':
    print 'Running ' + iii_1_1.__name__
    iii_1_1.run()
    
    continuePrompt()
    
    print 'Running ' + iii_2_1.__name__
    iii_2_1.run()

    continuePrompt()
    
    print 'Running ' + iii_2_2.__name__
    iii_2_2.run()
    
    continuePrompt()
    
    print 'Running ' + iii_2_3_1.__name__
    iii_2_3_1.run()
    