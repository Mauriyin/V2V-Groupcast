# V2V Communication:
Using DRL for the V2V groupcast resource allocation.

### Deep Reinforcement Learning design:
Action set construction   
observation: $o_n \in \{ACK,NACK\}$
state = {{o_i, a_i}, ...,{o_n, a_n}}
reward = 0 (NACK)/1(ACK)
