class Node:
    def __init__(self, index, depth=0):
        self.index = index
        self.depth = depth
        self.left_child = None
        self.right_child = None
        self.left_bound = None
        self.right_bound = None
        
    # iterative tree building
    def build_tree(root, depth):
        root.index = 0
        current_nodes = [root]

        counter = 1

        for i in range(0, depth):

            new_nodes = []

            for node in current_nodes:
                node.left_child = Node(counter, i+1)
                counter += 1
                node.right_child = Node(counter, i+1)
                counter += 1
                new_nodes.append(node.left_child)
                new_nodes.append(node.right_child)

            current_nodes = new_nodes.copy()

        return(root)


    def breadth_first(root):

        current_nodes = [root]

        while True:

            prev_nodes = current_nodes
            current_nodes = []

            for node in prev_nodes:
                print(node.index)
                current_nodes.append(node.left_child)
                current_nodes.append(node.right_child)

            if all(node is None for node in current_nodes):
                break
                
                
###################################
    def depth_first(root, max_depth=10):

        if (root == None):
            return

        st = []

        # start from root node (set current node to root node)
        curr = root
        curr_depth = 0
        # run till stack is not empty or current is
        # not NULL
        while (len(st) or curr != None):


            # Print left children while exist
            # and keep appending right into the
            # stack.
            while (curr != None):

                print(curr.index, end = " ")
                cutpoints[curr.index] = cutpoints[curr.index] + 2
                print(f"index:\t{curr.index}\tLB:\t{curr.left_bound}\tRB:\t{curr.right_bound}")

                if (curr.right_child != None):

                    curr.right_child.left_bound = cutpoints[curr.index]
                    curr.right_child.right_bound = curr.right_bound

                    if(curr_depth < max_depth):
                        st.append(curr.right_child)


                if (curr.left_child != None):
                    curr.left_child.left_bound = curr.left_bound
                    curr.left_child.right_bound = cutpoints[curr.index]

                if (curr_depth < max_depth):
                    curr = curr.left_child
                    curr_depth += 1
                else:
                    curr = None


            # We reach when curr is NULL, so We
            # take out a right child from stack
            if (len(st) > 0):

                curr = st[-1]
                st.pop()

###################################
            
            
            
    # Iterative function for inorder tree traversal
    def inorder(root, max_depth):

        # Set current to root of binary tree
        current = root
        current_depth = 0
        stack = [] # initialize stack
        out = []

        while True:

            # Reach the left most Node of the current Node
            if current is not None:

                # Place pointer to a tree node on the stack
                # before traversing the node's left subtree
                stack.append(current)
                current = current.left_child
                current_depth += 1


            # BackTrack from the empty subtree and visit the Node
            # at the top of the stack; however, if the stack is
            # empty you are done
            elif(stack):
                current = stack.pop()

                if current.depth <= max_depth:
                    out.append(current)

                # We have visited the node and its left
                # subtree. Now, it's right subtree's turn
                current = current.right_child

            else:
                break


        return(out)