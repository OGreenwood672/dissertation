import { sketch } from "./sketch";

const container = document.getElementById('sketch-container');

if (container) {
    new p5(sketch, container);
} else {
    console.log("Sketch Container not found");
}
