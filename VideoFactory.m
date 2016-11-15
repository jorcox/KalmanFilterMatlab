addpath RectGrabber/

movie = VideoWriter( sprintf( '%s\\results.avi', 'figures' ) );
open(movie)
for i=70:299
    A = imread(sprintf( '%s\\imgrect_%09d_c0.pgm', 'RectGrabber', i));
    writeVideo(movie,A)
    i
end
close(movie);