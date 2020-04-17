parallel do {
  {
    cook_dinner();
    eat_dinner();
  }
  {
    turn_on_radio();
    listen_to_music();
  }
} /* end of parallel do */
go_to_bed();
