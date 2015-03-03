import ii_1_1, ii_1_2

def continuePrompt():
    print 'Press ENTER to continue'
    raw_input()

if __name__ == '__main__':
    print 'Running ' + ii_1_1.__name__
    ii_1_1.run()
    
    continuePrompt()
    
    print 'Running ' + ii_1_2.__name__
    ii_1_2.run()

    continuePrompt()
    
    print 'Running ' + ii_2_1.__name__
    ii_2_1.run()
    
    continuePrompt()
    
    print 'Running ' + ii_2_2.__name__
    ii_2_2.run()
    