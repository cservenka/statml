import i_2_1, i_2_2, i_2_3, i_2_4, i_3_1, i_3_2, i_3_3

def continuePrompt():
    print 'Press ENTER to continue'
    raw_input()

if __name__ == '__main__':
    print 'Running ' + i_2_1.__name__
    i_2_1.run()
    
    continuePrompt()
    
    print 'Running ' + i_2_2.__name__
    (dataset, zValues) = i_2_2.run()
    
    continuePrompt()
    
    print 'Running ' + i_2_3.__name__
    i_2_3.run(dataset)
    
    continuePrompt()
    
    print 'Running ' + i_2_4.__name__
    i_2_4.run(dataset, zValues)
    
    continuePrompt()
    
    print 'Running ' + i_3_1.__name__
    i_3_1.run()
    
    continuePrompt()
    
    print 'Running ' + i_3_2.__name__
    i_3_2.run()
    
    continuePrompt()
    
    print 'Running ' + i_3_3.__name__
    i_3_3.run()