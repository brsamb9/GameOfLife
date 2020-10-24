extern crate piston_window;
use self::piston_window::*;
// https://github.com/PistonDevelopers/conrod/blob/master/backends/conrod_piston/examples/all_piston_window.rs

use std::time::{SystemTime, Duration};

mod game_of_life;
use game_of_life::{Ensemble, RandomGlider, Glider};

fn main() {
    const SCREEN_SIZE: u32 = 750;
    const SQUARE_SIZE: f64 = 10.0;
    const PIXEL_WIDTH: u32 = SCREEN_SIZE / SQUARE_SIZE as u32;
    const HISTORY_OF_STATES: usize = 5;
    const MILLISEC_PER_FRAME: u128 = 50;

    // Construct the window.
    let mut window: PistonWindow =
        WindowSettings::new("Mini-Project: Game of Life", [SCREEN_SIZE, SCREEN_SIZE])
            .graphics_api(OpenGL::V3_2) // If not working, try `OpenGL::V2_1`.
            .samples(1)
            .exit_on_esc(true)
            // .vsync(true)
            .build()
            .unwrap_or_else(|e| { panic!("Failed to build PistonWindow: {}", e) });

    // Just an binary array
    let mut ensemble_of_states = Ensemble::new((PIXEL_WIDTH * PIXEL_WIDTH) as usize, HISTORY_OF_STATES);

    let mut stall_time = SystemTime::now();

    while let Some(event) = window.next() {
        // Buggy behaviour w/ thread::sleep(time::Duration::from_millis(5)); [https://doc.rust-lang.org/std/thread/fn.sleep.html]
        // https://doc.rust-lang.org/std/time/struct.SystemTime.html

        // Next state is calculated after an imposed stall time
        // Not ideal as lots of redudant computation
        
        if stall_time.elapsed().unwrap_or(Duration::new(0, 0)).as_millis() > MILLISEC_PER_FRAME {
            stall_time = SystemTime::now();
            ensemble_of_states.next_state();    
            
            if let Some(mouse_position) = event.mouse_cursor_args() {
                let rescaled_mouse = [mouse_position[0] / SQUARE_SIZE, mouse_position[1] / SQUARE_SIZE];
                let new_ship = RandomGlider::assign(Glider::LightWeightSpaceShip, PIXEL_WIDTH, rescaled_mouse);

                for j in &new_ship.map {
                    ensemble_of_states.apply_life(*j as usize);
                }
            }
        } 

        window.draw_2d(&event, |context, graphics, _device| {
            clear(color::BLACK, graphics);
            // transformation matrix -> start at (0.0, 0.0) [top left]
            let context = context.trans(0.0, 0.0);
            
            for state in 0..HISTORY_OF_STATES {
                if let Some(elements_in_state) = &ensemble_of_states.0[state] {

                    for (i, pixel) in elements_in_state.0.iter().enumerate() {
                        if *pixel == 1 {
                            let row = i as u32 / PIXEL_WIDTH;
                            let col = i as u32 % PIXEL_WIDTH;
                            match state {
                                4 => rectangle(color::WHITE, [col as f64 * SQUARE_SIZE, row as f64 * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE], context.transform, graphics),
                                _ => rectangle(color::grey(state as f32 / 10.0), [col as f64 * SQUARE_SIZE, row as f64 * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE], context.transform, graphics),
                            }
                        }
                    }
                    
                }
            }
        });
        

    }
}
