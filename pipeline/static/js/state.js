// The one mutable object the views share. Everything in it is derived from
// /api/state (which the server reads off disk) or from the event stream, so a run
// launched from the CLI populates it exactly like one launched from the form.
export const state = {
  checksums: [],     // aggregate weight checksum, one per round
  vehicles: {},      // vid -> {received, sent, rounds, device, training}
  fleet: [],         // fleet.json summaries
  learning: null,    // vehicle_metrics.summary()
  cfg: {},           // the config the server is running
  current: null,     // stage running right now
  sel: null,         // vid whose drawer is open
  hidden: new Set(), // vids toggled off in the fleet chart legend
  drawer: null,      // {scrim, aside, vid}
};
