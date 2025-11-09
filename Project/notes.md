# Sim

Using rust for easy interaction with python and for performence reasons

## World

How to set out world to promote communication

Lots of different options
Pick up and drop off stations

### visualisation

Server/client for visualisation vs websockets.
Websockets maintains a connection, can just setup a rust thread to send the data every tick.
p5.js could not run correctly when simply importing, instead attached via script in html

## Agents
RNN vs LSTM
Learning VDN - make sure this is usable


variable items as input to neural net
- use padding
- use attention

Currently only looking at closest entity.
Maybe try and use attention later on

Max two items to to make one item in each agents processing.
This forces inventory to be max size one (as when second item picked up, immediately converted to output)