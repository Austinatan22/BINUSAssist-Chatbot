FROM node:22-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
# Empty default -> API calls go to the same origin the frontend is served from (relative
# paths like "/chat"), which is what docker-compose.prod.yml's Caddy reverse proxy expects.
# Override only if the backend is deployed on a separate origin/subdomain.
ARG VITE_API_URL=""
ENV VITE_API_URL=$VITE_API_URL
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
