## Question 2

A shepherd has several sheep, and every evening when he finishes herding, he just encloses all sheep in the sheepfold and then goes to bed. The sheepfold, which can be regarded as a polygon on the ground, is made up with many pieces of wooden fences. This morning the shepherd found that one of his sheep is missing. After examining, he found that a piece of fence is broken. And the unlucky sheep is possibly eaten by a wolf which came through the broken fence last night. The shepherd soon recognized that if he does not mend the sheepfold, he will lose a sheep every night from now on. However, since the broken piece cannot be used anymore, he has to reorganize the other pieces to make a new enclosure. As shorter fences are more likely to be broken (the broken piece is the shortest one), he decide to join some of the original fences to make longer fences, and make the new sheepfold as a rectangle, which is a shape with both long edges and large area. But since the original fences are all manufactured separately, they can only be joined at the ends. And if an original piece is divided into two or more pieces, all of them will be useless. Now with all the lengths of fences in the original sheepfold measured, the shepherd wonders how large the new sheepfold can be at most. (Sheep need large space to grow well) But he is only good at herding and do not know how to calculate it, can you help him?

### Input

Each test case begins with a line with only an integer N (3 <= N <= 17), the number of pieces in the original sheepfold. The next line contains N integers L1, L2 ... LN, (0 < Li <= 10,000, 1 <= i <= N), represent the length of each piece.
There are no more than 120 test cases in the input, processing to the end of file.

#### Sample

-   ```
    7
    1 1 3 3 4 5 7
    ```

-   ```
    7
    9 1 9 5 6 2 10
    ```

### Output

For each test case, print a line with the maximum area of the new sheepfold on its own. If it is impossible to make a valid sheepfold, print -1 instead.

#### Sample

-   ```
    15
    ```
-   ```
    -1
    ```
