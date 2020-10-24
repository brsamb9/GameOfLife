# Game of Life - https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life
Rust implementation of the famous game of life. 
<br><br>
Run `cargo run --release` in the directory to run the program. `esc` button to quit.
<br><br>
**Added features upon 'base' version:**
* Visualise previous live states as a decaying grey pixel.
* Move mouse to impose a lightweight space-ship in a random direction.
<br><br><br>
Random snap-shot:
![gol_snapshot](https://user-images.githubusercontent.com/69250411/97092474-17ea1e00-163c-11eb-9099-6fc8017ecfbb.png)
<br><br><br>
Other note-worthy mentions:
* Rust standard thread::sleep command was buggy, so lots of redudant computation was done.
