import * as p5 from "p5";

export const sketch = (p: p5) => {
  p.setup = () => {
    p.createCanvas(400, 400);
    p.background(230);
  };
  p.draw = () => {
    p.ellipse(p.mouseX, p.mouseY, 20, 20);
    console.log("Drawring");
  };
};
