/** MSW for Vitest (Node): the same handlers as the browser mocks. */
import { setupServer } from "msw/node";
import { handlers } from "./handlers";

export const server = setupServer(...handlers);
