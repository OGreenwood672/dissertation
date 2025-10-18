// import 'p5';
// import p5 from "p5";
import { sketch } from "./sketch";

const container = document.getElementById('sketch-container');

if (container) {
    console.log("2. Found the sketch container. Creating p5 instance...");
    new p5(sketch, container);
    console.log("3. p5 instance created.");
} else {
    console.log("Sketch Container not found");
}
